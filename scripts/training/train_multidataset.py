import os
import json
import sys
from pathlib import Path
# import torch
# import torch.distributed as dist
import numpy as np
import mindspore as ms
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch_geometric.loader import DataLoader
from sharker.loader.dataloader import Dataloader
from mindspore.communication import init, get_rank, get_group_size
# from torch.profiler import profile, record_function, ProfilerActivity


ROOT = Path(__file__).resolve().parents[2]
for _p in [str(ROOT), str(ROOT / "sharker")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- 导入自定义模块 (根据你的项目结构) ---
from src.data import ChunkedSmartDataset_h5, BinPackingSampler
from src.data.Dataset_dist import MultiSourceGraphDataset
from src.data.BinPackingSampler import MultiDatasetBinPackingSampler
from src.models import HTGPModel
from src.utils import HTGPConfig
from src.engine import PotentialTrainer 

# ==========================================
# 0. 全局环境设置 (Environment Setup)
# ==========================================

# 🚀 开启 TF32 (NVIDIA Ampere/Hopper 架构加速神器)
# torch.backends.cuda.matmul.allow_tf32 = True 
# torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. 训练配置 (Configuration)
# ==========================================
class Config:
    # 路径配置
    DATA_DIR = ["/home/qisong09/OMol25_dataset_h5", 
        "/home/qisong09/OMat24_dataset_h5", 
        "/home/qisong09/OMC_dataset_h5",
        "/home/qisong09/OC20_dataset_h5",
        "/home/qisong09/OC22_dataset_h5",
        "/home/qisong09/OC25_dataset_h5"]      # 数据根目录
    DATA_DIR_NAMES = ["OMol25", "OMat24", "OMC", "OC20", "OC22", "OC25"]
    # DATA_DIR_NAMES = ["OMol25", "OMat24", "OMC"]
    # DATA_DIR = ["/data0/chendanyang/UMA/OMat24_dataset_h5", 
    #     "/data0/chendanyang/UMA/OMC_dataset_h5"]      # 数据根目录
    # DATA_DIR_NAMES = ["OMat24", "OMC"]
    # RATIOS = {'OMat24':100, 'OMC':2, 'OMol25':100, 'OC20': 20, 'OC22': 10, 'OC25': 8}
    TRAIN_META = "train_metadata.pkl"         # 训练集元数据
    TEST_META = "test_metadata.pkl"           # 测试集元数据
    # E0_PATH = "meta_data.pt" # 原子能量参考值
    LOG_DIR = "Checkpoints"                  # 模型保存路径

    FINETUNE_MODE: bool = False  # <--- 总开关: True 为微调, False 为从头训练
    PRETRAINED_CKPT: str = "Checkpoints_pretrain/model_epoch_4.ckpt" # 旧模型路径

    # 断点续传配置
    RESUME: bool = False  # <--- True 时从 RESUME_CKPT 恢复训练
    RESUME_CKPT: str = "Checkpoints/model_epoch_2.ckpt"  # 续传的 checkpoint 路径

    # 每个 epoch 内保存几次 checkpoint (1 = 仅 epoch 结束时保存)
    SAVES_PER_EPOCH = 200

    # 训练超参
    # 🔥 注意: 这里的 BATCH_SIZE 指的是 "每个 Batch 的最大原子数 (Cost)"
    # 显存不足时可适当减小（如 600/512），会降低峰值显存、略增步数
    MAX_COST_PER_BATCH = 5000  # pretrain 20000, finetune 3500
    LR = 1e-3 # pretrain 2e-3, finetune 4e-4
    EPOCHS = 1
    
    # 系统配置
    NUM_WORKERS = 8            # DataLoader 进程数
    PREFETCH_FACTOR = 2        # 预取因子（显存/内存紧张可改为 1）

    # 模型配置 (HTGP)
    MODEL_PARAMS = dict(
        hidden_dim=128,
        num_layers=2,
        cutoff=6.0,
        num_rbf=32,
        use_L0=True,
        use_L1=True,
        use_L2=True,
        use_gating=True,
        use_long_range=False,
        use_recompute=False,   # 开启重计算：反向时重算激活，节省显存
    )

# ==========================================
# 2. 辅助函数 (Utils)
# ==========================================
def init_distributed_mode():
    """初始化 DDP 分布式环境"""
    parallel_mode = (
        os.environ.get("PARALLEL_MODE", "NONE").upper()
    )
    if parallel_mode == "DATA_PARALLEL":
        # rank = int(os.environ["RANK"])
        # world_size = int(os.environ["WORLD_SIZE"])
        # local_rank = int(os.environ["LOCAL_RANK"])
        
        # # torch.cuda.set_device(local_rank)
        # dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        # dist.barrier()
        init()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        rank_id = get_rank()
        rank_size = get_group_size()
        return rank_id, rank_size
    else:
        print("⚠️ Warning: Running in Single NPU Mode")
        return 0, 1

def log_info(msg, rank):
    """仅在主进程打印日志"""
    if rank == 0:
        print(msg)

def get_dataloader(data_dir, meta_file, rank, world_size, is_train=True):
    """构建 Dataset, Sampler 和 DataLoader"""
    datasets = {}
    for ds_name, ds in zip(Config.DATA_DIR_NAMES, data_dir):
        full_path = os.path.join(ds, ds_name + "_" + meta_file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"❌ 致命错误: 没找到 {meta_file}，请先运行 preprocess.py！")
        datasets[ds_name] = ChunkedSmartDataset_h5(
            ds, 
            metadata_file=ds_name + "_" + meta_file, 
            rank=rank,
            world_size=world_size
        )
    router_ds = MultiSourceGraphDataset(datasets)

    # 2. Sampler (训练用 Shuffle, 测试不用)
    sampler = MultiDatasetBinPackingSampler(
        {x:datasets[x].metadata for x in datasets.keys()},
        max_cost=Config.MAX_COST_PER_BATCH,
        edge_weight="auto",
        shuffle=is_train,
        world_size=world_size,
        rank=rank
    )

    # 3. Loader
    loader = Dataloader(
        router_ds,
        sampler=sampler, # 关键：使用 batch_sampler 处理动态 Batch
        num_parallel_workers=Config.NUM_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
        # 注意：BinPackingSampler 已经做了 world_size/rank 切片，这里不要再 num_shards/shard_id 二次切片
    )
    # loader = loader.batch(2)
    
    return loader, sampler

def build_model(rank, avg_neighborhood, **karwgs):
    """构建模型并加载 E0"""
    # 初始化配置和模型
    model_config = HTGPConfig(**Config.MODEL_PARAMS)
    model_config.avg_neighborhood = avg_neighborhood
    model = HTGPModel(model_config)
    if "restart" in karwgs:
        state_dict = karwgs["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v

        param_not_load, _ = ms.load_param_into_net(model, new_state_dict)
        if param_not_load:
            print(f"❌ Warning: {param_not_load} parameters not loaded!")
        else:
            print("✅ All parameters loaded successfully!")

    # bf16: 将网络计算 dtype 切换为 bfloat16（loss 建议保持 fp32，在 trainer 里做）
    if getattr(model_config, "use_bf16", False):
        if rank == 0:
            log_info("🧮 Enabling BF16 compute: model.to_float(ms.bfloat16)", rank)
        model.to_float(ms.bfloat16)
    
    # 打印参数量
    if rank == 0:
        param_count = sum(p.numel() for p in model.get_parameters())
        log_info(f"🧠 Model Parameters: {param_count:,}", rank)


    # # 注入 E0 (原子参考能量)
    # if "restart" not in karwgs:
    #     e0_dicts = None
        
    #     import pickle
    #     e0_dicts = {}
    #     for ds_name, ds in zip(Config.DATA_DIR_NAMES, Config.DATA_DIR):
    #         # E0 的文件路径按约定：DATA_DIR + E0_PATH（相对路径则拼接 DATA_DIR）
    #         e0_path = os.path.join(ds, Config.E0_PATH)
    #         if os.path.exists(e0_path):
    #             with open(e0_path, "rb") as f:
    #                 meta = pickle.load(f)
    #             e0_dicts[ds_name] = meta.get("e0_dict", meta)
    #     if e0_dicts:
    #         model.load_external_e0_multi(e0_dicts, verbose=(rank == 0), rank=rank)
    #     model.atomic_ref.embedding_table.requires_grad = False # 冻结 E0
    # # DDP 包装
    # if dist.is_initialized():
    #     model = DDP(model, find_unused_parameters=True)
    
    return model

# ==========================================
# 3. 主程序 (Main)
# ==========================================
def main():
    # ms.set_context(save_graphs=True, save_graphs_path='./ir')
    # --- A. 初始化环境 ---
    rank, world_size = init_distributed_mode()
    # device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        log_info(f"\n🚀 [Start] World Size: {world_size}", rank)
        log_info("="*60, rank)

    # --- B. 准备数据 ---
    log_info("\n[1/4] Initializing DataLoaders...", rank)
    
    # 训练集
    train_loader, train_sampler = get_dataloader(
        Config.DATA_DIR, Config.TRAIN_META, rank, world_size, is_train=True
    )

    # 测试集 
    test_loader, test_sampler = get_dataloader(
        Config.DATA_DIR, Config.TEST_META, rank, world_size, is_train=False
    )
    # 这里的解包逻辑稍微改一下，防止 test_result 为 None 报错
    # test_loader = test_result[0] if test_result else None

    # --- C. 构建模型 ---
    log_info("\n[2/4] Building Model...", rank)
    
    avg_neighborhood = 1 / train_sampler.edge_weight
    if not Config.FINETUNE_MODE:
        
        model = build_model(rank, avg_neighborhood)
    else:
        checkpoint_path = Config.PRETRAINED_CKPT
        checkpoint_weights = ms.load_checkpoint(checkpoint_path)

        model = build_model(rank, avg_neighborhood, restart=Config.FINETUNE_MODE, state_dict=checkpoint_weights)

    # --- D. 初始化 Trainer ---
    log_info("\n[3/4] Initializing Trainer...", rank)
    
    # 估算总步数 (因为是动态 Batch，步数不是固定的 len/bs，需要从 sampler 获取)
    train_total_steps = train_sampler.precompute_total_steps(Config.EPOCHS)
    log_info(f"📊 Estimated total steps for training: {train_total_steps}", rank)

    # 🔥 修改 2: 必须加 if 判断，否则 test_sampler 为 None 时会报错
    if test_sampler is not None:
        test_total_steps = test_sampler.precompute_total_steps(Config.EPOCHS)
        log_info(f"📊 Estimated total steps for testing: {test_total_steps}", rank)

    
    trainer = PotentialTrainer(
    model,
    total_steps=train_total_steps,
    max_lr=Config.LR,
    checkpoint_dir=Config.LOG_DIR,
    saves_per_epoch=Config.SAVES_PER_EPOCH)

    # --- D2. 断点续传 ---
    start_epoch = 1
    resume_skip_steps = 0  # epoch 内需要跳过的步数
    # 计算当前 world_size 下每 epoch 的步数，用于跨卡数续传换算
    cur_steps_per_epoch = train_total_steps // Config.EPOCHS
    if Config.RESUME:
        resume_ckpt = os.path.basename(Config.RESUME_CKPT)
        trainer.checkpoint_dir = os.path.dirname(Config.RESUME_CKPT) or Config.LOG_DIR
        resumed_epoch, intra_step = trainer.load_checkpoint(resume_ckpt, new_steps_per_epoch=cur_steps_per_epoch)
        trainer.checkpoint_dir = Config.LOG_DIR  # 恢复保存目录
        if intra_step is not None:
            # epoch 内中断：从同一 epoch 继续，跳过已完成的步数
            start_epoch = resumed_epoch
            resume_skip_steps = intra_step
            log_info(f"🔄 Resuming epoch {start_epoch} from step {resume_skip_steps}", rank)
        else:
            # epoch 完整结束：从下一个 epoch 开始
            start_epoch = resumed_epoch + 1
            log_info(f"🔄 Resuming training from epoch {start_epoch}", rank)

    # --- E. 训练循环 ---
    log_info("\n[4/4] Starting Loop...", rank)
    log_info("="*60, rank)


    for epoch in range(start_epoch, Config.EPOCHS + 1):
        # 重要：每个 Epoch 设置随机种子，保证 Shuffle 效果
        train_sampler.set_epoch(epoch)

        # 1. Train (传入 skip_steps 跳过已完成的步)
        skip = resume_skip_steps if epoch == start_epoch else 0
        train_metrics = trainer.train_epoch(train_loader, epoch_idx=epoch, skip_steps=skip, steps_per_epoch=cur_steps_per_epoch)

        # 2. Validate
        if test_loader:
            val_metrics = trainer.validate(test_loader, epoch_idx=epoch)
        else:
            val_metrics = {'total_loss': 0.0, 'mae_f': 0.0}

        # 3. Log & Save (仅 Rank 0)
        if rank == 0:
            log_msg = (
                f"Ep {epoch:03d} | "
                f"T_Loss: {train_metrics['total_loss']:.4f} | "
                f"V_Loss: {val_metrics['total_loss']:.4f} | "
                f"MAE_F: {train_metrics['mae_f']*1000:.1f}/{val_metrics['mae_f']*1000:.1f} meV/A"
            )
            print(log_msg)
            trainer.save(f'model_epoch_{epoch}.ckpt', epoch=epoch, val_metrics=val_metrics, steps_per_epoch=cur_steps_per_epoch)

        # # 4. 同步：确保所有卡都跑完了这个 Epoch
        # if dist.is_initialized():
        #     dist.barrier()

    log_info("\n✅ Training Finished!", rank)
    
    # # --- F. 清理 ---
    # if dist.is_initialized():
    #     dist.destroy_process_group()

if __name__ == "__main__":
    # 设置 OMP 线程数，防止 CPU 过载
    os.environ["OMP_NUM_THREADS"] = "1" 
    main()
