import mindspore as ms
from mindspore import nn, mint, ops
from mindspore import Tensor, Parameter
import os
import csv
import pickle
from tqdm.auto import tqdm
from src.utils import scatter_add
import mindspore.communication as dist
from mindspore.communication import get_group_size
from mindspore.experimental.optim import AdamW
import numpy as np
from src.utils.scheduler import OneCycleLR

# MindSpore JIT 编译方式
# @ms.jit
def conditional_huber_loss(pred: Tensor, target: Tensor, base_delta: float = 0.01) -> Tensor:
    """
    自适应 Huber Loss (MindSpore JIT Optimized)
    """
    # bf16 兼容性/数值稳定性：loss 相关计算强制走 fp32
    pred = pred.astype(ms.float32)
    target = target.astype(ms.float32)

    # 计算每个原子的受力模长 (N, 1)
    force_norm = ms.mint.norm(target, dim=1, keepdim=True)
    
    # 初始化缩放因子
    delta_scale = ms.mint.ones_like(force_norm)
    
    # 阶梯式降级策略
    mask_100_200 = ms.mint.logical_and(force_norm >= 100, force_norm < 200)
    delta_scale = ops.select(mask_100_200, Tensor(0.7, ms.float32), delta_scale)
    
    mask_200_300 = ms.mint.logical_and(force_norm >= 200, force_norm < 300)
    delta_scale = ops.select(mask_200_300, Tensor(0.4, ms.float32), delta_scale)
    
    mask_300 = (force_norm >= 300)
    delta_scale = ops.select(mask_300, Tensor(0.1, ms.float32), delta_scale)
    
    # 计算最终的 delta
    adaptive_delta = base_delta * delta_scale
    
    # 手动实现 Huber 计算逻辑
    error = pred - target
    abs_error = ms.mint.abs(error)
    
    # 判定 MSE 区域
    is_mse = abs_error < adaptive_delta
    
    loss_mse = 0.5 * error.astype(ms.float32) ** 2
    loss_l1 = adaptive_delta * (abs_error - 0.5 * adaptive_delta)
    
    # 组合并取平均
    loss = ops.select(is_mse, loss_mse, loss_l1)
    return loss.mean()


class ExponentialMovingAverage:
    """MindSpore EMA 实现（torch_ema 的替代）"""

    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self._parameters = list(parameters)
        self.shadow_params = [
            p.data.copy() if p.requires_grad else None
            for p in self._parameters
        ]
        self.collected_params = []

    def update(self, parameters=None):
        """更新 shadow 参数；无参时默认使用初始化时保存的参数引用"""
        if parameters is None:
            parameters = self._parameters
        for i, (s_param, param) in enumerate(zip(self.shadow_params, parameters)):
            if s_param is not None and param.requires_grad:
                self.shadow_params[i] = self.decay * s_param + (1.0 - self.decay) * param.data

    def average_parameters(self):
        """上下文管理器：验证/保存时临时将模型参数替换为 EMA 权重"""
        return self._AvgContext(self)

    class _AvgContext:
        def __init__(self, ema):
            self.ema = ema

        def __enter__(self):
            ema = self.ema
            ema.collected_params = []
            for param, s_param in zip(ema._parameters, ema.shadow_params):
                if s_param is not None:
                    ema.collected_params.append(param.data.copy())
                    param.set_data(s_param)
                else:
                    ema.collected_params.append(None)
            return self

        def __exit__(self, *_):
            ema = self.ema
            for param, c_param in zip(ema._parameters, ema.collected_params):
                if c_param is not None:
                    param.set_data(c_param)


class PotentialTrainer:
    def __init__(self,
                 model,
                 total_steps,
                 max_lr=1e-3,
                 device='Ascend',  # MindSpore 使用 'Ascend', 'GPU', 'CPU'
                 checkpoint_dir='checkpoints',
                 finetune_mode=False,
                 saves_per_epoch=1,
                 **kwargs):
        """
        MindSpore 版本的 PotentialTrainer

        Args:
            total_steps: 总训练步数
            epochs: 总训练轮次
            device: 'Ascend', 'GPU', 或 'CPU'
            saves_per_epoch: 每个 epoch 内保存几次 checkpoint (默认 1 = 仅 epoch 结束时保存)
        """

        self.device = device
        self.model = model
        self.finetune_mode = finetune_mode
        self.global_step = 0
        self.saves_per_epoch = saves_per_epoch
        # ⚠️ MindSpore 分布式获取 rank 的方式不同
        try:
            self.rank = dist.get_rank()
        except RuntimeError:
            self.rank = 0
            
        self.checkpoint_dir = checkpoint_dir

        if kwargs.get("only_les", False):
            for param in self.model.trainable_params():
                param_name = param.name
                if "long_range" in param_name or "sigma" in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # 1. 优化器配置
        if self.finetune_mode:
            if self.rank == 0:
                print(f"[Trainer] Initializing in FINETUNE mode (lr={max_lr})")

            all_params = [p for p in self.model.trainable_params() if p.requires_grad]
            if not all_params:
                raise ValueError("No trainable parameters found for fine-tuning.")
            self.optimizer = AdamW(
                [{'params': all_params, 'lr': max_lr, 'weight_decay': 1e-2}],
                lr=max_lr,
                betas=(0.9, 0.999),
            )
            self.scheduler = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=max_lr,
                total_steps=int(total_steps * 1.02),
                pct_start=0.03,
                anneal_strategy="cos",
                div_factor=5.0,
                final_div_factor=100.0,
                three_phase=False,
            )

        else:
            # ---> 分支 B: 原始从头训练模式 <---
            dataset_emb_params = []
            other_params = []
            for param in self.model.trainable_params():
                if "dataset_embedding" in param.name:
                    dataset_emb_params.append(param)
                else:
                    other_params.append(param)

            param_groups = [
                {'params': other_params, 'weight_decay': 1e-3},
                {'params': dataset_emb_params, 'weight_decay': 0.01},
            ]
            self.optimizer = AdamW(
                param_groups,
                lr=max_lr,
                betas=(0.9,0.999)
            )
            self.scheduler = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=max_lr,
                total_steps=int(total_steps * 1.02),
                pct_start=0.01,
                anneal_strategy="cos",
                div_factor=5.0,
                final_div_factor=200.0,
                three_phase=False,
            )

        # 2. EMA (指数移动平均) - 使用自定义实现
        # finetune 数据量小、步数少，用更低的 decay 让 EMA 跟得上权重变化
        ema_decay = 0.99 if finetune_mode else 0.999
        self.ema = ExponentialMovingAverage(self.model.trainable_params(), decay=ema_decay)

        # 3. 学习率调度器
        # ⚠️ MindSpore 学习率调度器的使用方式完全不同
        # MindSpore 需要在创建优化器时传入学习率调度器，而不是后续创建
        self.finetune_mode_flag = finetune_mode
        self.max_lr = max_lr
        self.total_steps = total_steps

        # Loss 配置
        self.huber_delta = 0.01
        self.w_e = 10.0
        self.w_f = 100.0
        self.w_s = 10.0
        
        self.train_log_path = os.path.join(self.checkpoint_dir, 'train_log.csv')
        self.val_log_path = os.path.join(self.checkpoint_dir, 'val_log.csv')
        self.EV_A3_TO_GPA = 160.21766 
        
        # 日志：延迟初始化，避免 resume 时覆盖已有日志
        self._loggers_initialized = False
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.grad_fn = ms.value_and_grad(self.step, None, self.optimizer.parameters, has_aux=True)
        self.parallel_mode = (
            os.environ.get("PARALLEL_MODE", "NONE").upper()
        )
        if self.parallel_mode == "DATA_PARALLEL":
            self.grad_reducer = nn.DistributedGradReducer(
                self.optimizer.parameters, mean=True, degree=get_group_size())

    def _get_world_size(self):
        try:
            return get_group_size()
        except RuntimeError:
            return 1

    def _init_loggers(self, resume=False):
        headers = ['epoch', 'step', 'lr', 'total_loss', 'loss_e', 'loss_f', 'loss_s', 'mae_e', 'mae_f', 'mae_s_gpa']
        for path in [self.train_log_path, self.val_log_path]:
            if resume and os.path.exists(path):
                continue  # 续传时保留已有日志
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(headers)

    def log_to_csv(self, mode, data):
        if self.rank != 0:
            return
        # 懒初始化：首次写日志时才创建文件头（resume 时文件已存在则跳过）
        if not self._loggers_initialized:
            self._init_loggers(resume=True)
            self._loggers_initialized = True
        path = self.train_log_path if mode == 'train' else self.val_log_path
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow([
                data['epoch'], data['step'], f"{data['lr']}",
                f"{data['total_loss']}", f"{data['loss_e']}",
                f"{data['loss_f']}", f"{data['loss_s']}",
                f"{data['mae_e']*1000}", f"{data['mae_f']*1000}", f"{data['mae_s_gpa']}"
            ])

    def _current_lr(self, group_idx=-1):
        lr = self.optimizer.param_groups[group_idx]['lr']
        if isinstance(lr, Parameter):
            lr = lr.value()
        if isinstance(lr, ms.Tensor):
            return float(lr.asnumpy())
        return float(lr)

    def step(self, batch, train=True, batch_idx=0):
        """
        ⚠️ 警告: MindSpore 的自动微分机制与 PyTorch 有较大差异
        特别是 torch.autograd.grad 的功能在 MindSpore 中需要使用不同的API实现
        """
        # ⚠️ MindSpore 的 requires_grad 设置方式不同
        # batch.pos.requires_grad_(True)  # 这在 MindSpore 中不适用
        
        use_direct_force = self.model.cfg.use_direct_force

        # 获取图数量
        if hasattr(batch, 'num_graphs'):
            num_graphs = batch.num_graphs
        else:
            num_graphs = int(ops.reduce_max(batch.batch if batch.batch is not None else ms.Tensor(1))) + 1

        original_pos = batch.pos
        original_cell = getattr(batch, 'cell', None)

        # ============================================================
        # Path A: use_direct_force=True
        # 直接从 model 输出读取 energy/force，不计算 stress，loss 也不加入 stress
        # ============================================================
        if use_direct_force:
            out = self.model(batch)
            if isinstance(out, dict):
                pred_e = out.get('energy', None)
                if pred_e is None:
                    raise ValueError("use_direct_force=True but model(batch) did not return key 'energy'.")
                pred_e = pred_e.view(-1)
                pred_f = out.get('force', mint.zeros_like(batch.pos))
            else:
                # 兼容：如果 model 仍然只返回能量 tensor
                pred_e = out.view(-1)
                pred_f = mint.zeros_like(batch.pos)

            pred_stress = None

        # ============================================================
        # Path B: use_direct_force=False
        # 通过能量对 pos/strain 求导得到 force & stress
        # ============================================================
        else:
            displacement = mint.zeros((num_graphs, 3, 3)).astype(ms.float32)

            def get_energy(pos, disp):
                # --- 构造虚拟应变 ---
                symmetric_strain = 0.5 * (disp + ops.transpose(disp, (0, 2, 1)))

                # --- 应用变形 ---
                strain_per_atom = symmetric_strain[batch.batch]
                pos_deformed = pos + mint.einsum('ni,nij->nj', pos, strain_per_atom)

                batch.pos = pos_deformed
                if original_cell is not None and len(original_cell.shape) == 3:
                    batch.cell = original_cell + ops.matmul(original_cell, symmetric_strain)

                # --- 前向传播 ---
                # bf16 兼容性：用于求导的能量值也强制 fp32，避免部分算子/梯度链不支持 bf16
                pred_e_inner = self.model(batch).view(-1).astype(ms.float32)
                return pred_e_inner, original_cell

            grads_fn = ms.value_and_grad(get_energy, grad_position=(0, 1), has_aux=True)
            (pred_e, _), grads = grads_fn(original_pos, displacement)

            # 恢复 batch
            batch.pos = original_pos
            if original_cell is not None:
                batch.cell = original_cell

            pred_f = -grads[0] if grads[0] is not None else mint.zeros_like(batch.pos)
            dE_dStrain = grads[1]

            # --- 体积计算与安全除法 ---
            if dE_dStrain is not None:
                if original_cell is not None:
                    vol = mint.abs(mint.exp(ops.logdet(original_cell).astype(ms.float32))).view(-1, 1, 1)
                else:
                    vol = mint.ones((num_graphs, 1, 1), ms.float32)
                pred_stress = (dE_dStrain.astype(ms.float32)) / vol
            else:
                pred_stress = mint.zeros((num_graphs, 3, 3)).astype(ms.float32)
        
        
        # ==================================================================
        # Loss 计算
        # ==================================================================
        # bf16 兼容性：loss/metric 统一用 fp32
        target_e = batch.y.view(-1).astype(ms.float32)
        pred_e = pred_e.astype(ms.float32)
        pred_f = pred_f.astype(ms.float32)
        if pred_stress is not None:
            pred_stress = pred_stress.astype(ms.float32)
        if hasattr(batch, 'force') and batch.force is not None:
            batch.force = batch.force.astype(ms.float32)
        if hasattr(batch, 'stress') and batch.stress is not None:
            batch.stress = batch.stress.astype(ms.float32)
        
        # 计算每个图的原子数
        if not hasattr(self, '_ones_buffer') or self._ones_buffer.shape[0] != batch.batch.shape[0]:
            self._ones_buffer = mint.ones(batch.batch.shape).astype(ms.float32)
        
        num_atoms = scatter_add(self._ones_buffer, batch.batch, dim=0, dim_size=num_graphs).view(-1).clamp(min=1)
        
        # ⚠️ MindSpore 的 SmoothL1Loss (Huber Loss)
        loss_e = ops.huber_loss(pred_e / num_atoms, target_e / num_atoms, delta=self.huber_delta)
        
        # 使用自定义的条件 Huber Loss
        loss_f = conditional_huber_loss(pred_f, batch.force, base_delta=self.huber_delta)
        
        loss_s = Tensor(0.0, ms.float32)
        stress_mask_sum = 0

        if (not use_direct_force) and hasattr(batch, 'stress') and batch.stress is not None:
            stress_norm = mint.norm(batch.stress.view(num_graphs, -1), dim=1)
            stress_mask = (stress_norm > 1e-6)

            # 根据 stress_datasets 配置，按数据集过滤不需要计算 stress 的图
            cfg = self.model.module.cfg if hasattr(self.model, 'module') else self.model.cfg
            if hasattr(batch, 'dataset') and batch.dataset is not None and hasattr(cfg, 'stress_datasets'):
                # dataset_types: {'OMol25': 0, 'OMat24': 1, ...} -> 反向映射 {0: 'OMol25', ...}
                idx_to_name = {v: k for k, v in cfg.dataset_types.items()}
                # 构造每个图是否启用 stress 的 bool 列表
                ds_stress_flags = [
                    cfg.stress_datasets.get(idx_to_name.get(int(d), '') if str(d).isdigit() else str(d), False)
                    for d in batch.dataset
                ]
                ds_stress_mask = ms.Tensor(ds_stress_flags, dtype=ms.bool_)
                stress_mask = stress_mask & ds_stress_mask

            stress_mask_sum = stress_mask.sum().item()

            if stress_mask_sum > 0:
                s_pred = pred_stress.view(num_graphs, -1)[stress_mask]
                s_target = batch.stress.view(num_graphs, -1)[stress_mask]
                loss_s = ops.huber_loss(s_pred, s_target, delta=self.huber_delta)

        total_loss = self.w_e * loss_e + self.w_f * loss_f
        if not use_direct_force:
            total_loss = total_loss + self.w_s * loss_s
        
        # --- Metrics 计算 ---
        with ms._no_grad():
            mae_e = ops.reduce_mean(ops.abs(pred_e / num_atoms - target_e / num_atoms)).item()
            mae_f = ops.reduce_mean(ops.abs(pred_f - batch.force)).item()
            mae_s_gpa = 0.0
        
        if (not use_direct_force) and stress_mask_sum > 0:
            mae_s_val = ops.reduce_mean(
                ops.abs(pred_stress.view(num_graphs, -1)[stress_mask] - 
                       batch.stress.view(num_graphs, -1)[stress_mask])
            )
            mae_s_gpa = mae_s_val.item() * self.EV_A3_TO_GPA
        
        return total_loss, {
            'total_loss': total_loss.asnumpy().item(),
            'loss_e': loss_e.asnumpy().item(), 
            'loss_f': loss_f.asnumpy().item(), 
            'loss_s': loss_s.asnumpy().item(),
            'mae_e': mae_e, 
            'mae_f': mae_f, 
            'mae_s_gpa': mae_s_gpa
        }

    def _log_gradients(self, grads, step, epoch):
        """诊断关键参数的梯度，仅在指定步打印，避免日志刷屏。"""
        params = list(self.optimizer.parameters)

        # ---- 1. 全局梯度 norm（裁剪前） ----
        sq_sum = ms.Tensor(0.0, ms.float32)
        has_nan = False
        for g in grads:
            if g is None:
                continue
            gf = g.astype(ms.float32)
            nan_flag = not bool(mint.isfinite(gf).all().asnumpy())
            if nan_flag:
                has_nan = True
            sq_sum = sq_sum + mint.sum(gf * gf)
        global_norm = float(mint.sqrt(sq_sum).asnumpy())

        print(f"\n{'='*60}")
        print(f"[GradLog] Epoch {epoch}, Step {step}  |  Global Grad Norm (pre-clip): {global_norm:.4f}  |  NaN/Inf: {has_nan}")
        print(f"{'='*60}")

        # ---- 2. 所有参数逐一打印 ----
        header = f"{'Parameter':<60} {'Norm':>10} {'Max':>10} {'Min':>10} {'NaN':>5}"
        print(header)
        print('-' * len(header))

        for param, grad in zip(params, grads):
            name = param.name
            if grad is None:
                print(f"{name:<60} {'None':>10}")
                continue
            gf = grad.astype(ms.float32)
            norm_val  = float(mint.norm(gf).asnumpy())
            max_val   = float(gf.max().asnumpy())
            min_val   = float(gf.min().asnumpy())
            nan_flag  = not bool(mint.isfinite(gf).all().asnumpy())
            print(f"{name[-60:]:<60} {norm_val:>10.4e} {max_val:>10.4e} {min_val:>10.4e} {str(nan_flag):>5}")

        print(f"{'='*60}\n")

    def train_epoch(self, loader, epoch_idx, skip_steps=0, steps_per_epoch=None, on_step_fn=None):
        self.model.set_train(True)
        pbar = tqdm(loader, desc=f"Train Ep {epoch_idx}", leave=False, disable=(self.rank != 0))
        metrics_sum = {'mae_e': 0, 'mae_f': 0, 'mae_s_gpa': 0, 'total_loss': 0}
        count = 0

        max_steps = getattr(self.model.module.cfg, 'steps_per_epoch', None) if hasattr(self.model, 'module') else getattr(self.model.cfg, 'steps_per_epoch', None)

        if skip_steps > 0 and self.rank == 0:
            print(f"   ⏩ Skipping first {skip_steps} steps (already completed)")

        for i, batch in enumerate(pbar):
            # 断点续传：跳过已完成的步
            if i < skip_steps:
                continue

            if i == skip_steps:
                if self.rank == 0:
                    print("First batch graph info:")
                    print("Number of graphs in batch:", batch.num_graphs)
                    print("Nodes (atoms) in batch:", batch.pos.shape[0])
                    print("Edge index:", batch.edge_index)
                    print("Batch indices:", batch.batch)
                    if hasattr(batch, 'stress') and batch.stress is not None:
                        print("Stress tensor shape:", batch.stress.shape)
                    else:
                        print("No stress tensor in this batch.")

            # metrics = self.step(batch, train=True, batch_idx=i)

            (_, metrics), grads = self.grad_fn(batch, train=True, batch_idx=i)
            if self.parallel_mode == "DATA_PARALLEL":
                grads = self.grad_reducer(grads)
            # 梯度诊断：前 3 步打印（裁剪前），仅 rank 0
            if self.rank == 0 and i < 3:
                self._log_gradients(grads, step=i, epoch=epoch_idx)
            grads = ops.clip_by_global_norm(grads, clip_norm=1.0)
            self.optimizer(grads)
                # total_loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # self.optimizer.step()
                
            # finetune 模式每步都更新 EMA（数据量小、步数少）；预训练每 5 步一次
            if self.finetune_mode or i % 5 == 0:
                self.ema.update()  
            
            # 记录 CS
            self.global_step += 1
            log_data = metrics.copy()
            if self.finetune_mode:
                log_data.update({'epoch': epoch_idx, 'step': i, 'lr': self._current_lr()})
            else:
                log_data.update({'epoch': epoch_idx, 'step': i, 'lr': self._current_lr(0)})

            self.log_to_csv('train', log_data)
            self.scheduler.step()
            # 统计
            for k in metrics_sum:
                metrics_sum[k] += metrics[k]
            count += 1
            pbar.set_postfix({
                'L': f"{metrics['total_loss']:.4f}",
                'MAE_e': f"{metrics['mae_e']*1000:.1f}",
                'MAE_F': f"{metrics['mae_f']*1000:.1f}"
            })

            if on_step_fn is not None:
                on_step_fn(epoch_idx, i + 1, metrics)

            # epoch 内定期保存 checkpoint
            if self.saves_per_epoch > 1 and self.rank == 0:
                total = max_steps if max_steps is not None else getattr(loader, '__len__', lambda: None)()
                if total is not None:
                    save_interval = max(1, total // self.saves_per_epoch)
                    if (i + 1) % save_interval == 0 and (i + 1) < total:
                        ckpt_name = f'model_epoch_{epoch_idx}_step_{i+1}.ckpt'
                        self.save(ckpt_name, epoch=epoch_idx, intra_epoch_step=i+1, steps_per_epoch=steps_per_epoch)
                        print(f"   💾 Intra-epoch checkpoint saved at step {i+1}/{total}")

            if max_steps is not None and (i + 1) >= max_steps:
                if self.rank == 0:
                    print(f"   🛑 Virtual Epoch Reached ({max_steps} steps). Stopping for Validation...")
                break

        return {k: v/count for k,v in metrics_sum.items()}

    def validate(self, loader, epoch_idx):
        """验证循环"""
        self.model.set_train(False)
        pbar = tqdm(loader, desc=f"Val Ep {epoch_idx}", leave=False, disable=(self.rank != 0))
        metrics_sum = {'mae_e': 0, 'mae_f': 0, 'mae_s_gpa': 0, 'total_loss': 0}
        count = 0
        
        max_steps = getattr(self.model.module.cfg, 'steps_per_epoch', None) if hasattr(self.model, 'module') else getattr(self.model.cfg, 'steps_per_epoch', None)

        with self.ema.average_parameters():
            # with torch.set_grad_enabled(True):
            
            for i, batch in enumerate(pbar):
                
                metrics = self.step(batch, train=False)[1]
                
                log_data = metrics.copy()
                if self.finetune_mode:
                    log_data.update({'epoch': epoch_idx, 'step': i, 'lr': self._current_lr()})
                else:
                    log_data.update({'epoch': epoch_idx, 'step': i, 'lr': self._current_lr(0)})

                self.log_to_csv('val', log_data)
                
                for k in metrics_sum: metrics_sum[k] += metrics[k]
                count += 1
                pbar.set_postfix({'L': f"{metrics['total_loss']:.4f}", 
                                    'MAE_e': f"{metrics['mae_e']*1000:.1f}",
                                    'MAE_F': f"{metrics['mae_f']*1000:.1f}"})
                if max_steps is not None and (i + 1) >= max_steps:
                    if self.rank == 0:
                        print(f"   🛑 Virtual Epoch Reached ({max_steps} steps). Stopping for training...")
                    break 
        
        if count == 0: 
            count = 1
        return {k: v/count for k,v in metrics_sum.items()}

    def step_scheduler_on_val(self, val_loss):
        if self.finetune_mode and hasattr(self, 'scheduler'):
            self.scheduler.step(val_loss)

    def _serialize_optimizer_state(self):
        """将 optimizer.state 序列化为 {param_name: {key: ndarray}} 格式"""
        if not hasattr(self.optimizer, 'state'):
            return {}
        # 建立 param object -> name 的映射
        param_to_name = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_to_name[id(p)] = p.name
        result = {}
        for param, state_dict in self.optimizer.state.items():
            name = param_to_name.get(id(param), None)
            if name is None:
                continue
            result[name] = {
                sk: sv.asnumpy() if isinstance(sv, (ms.Tensor, Parameter)) else sv
                for sk, sv in state_dict.items()
            }
        return result

    def _restore_optimizer_state(self, saved_state):
        """从 {param_name: {key: ndarray}} 恢复 optimizer.state"""
        if not saved_state or not hasattr(self.optimizer, 'state'):
            return
        # 建立 name -> param object 的映射
        name_to_param = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                name_to_param[p.name] = p
        for param_name, state_dict in saved_state.items():
            param = name_to_param.get(param_name)
            if param is None:
                continue
            if param not in self.optimizer.state:
                self.optimizer.state[param] = {}
            for sk, sv in state_dict.items():
                if isinstance(sv, np.ndarray):
                    self.optimizer.state[param][sk] = ms.Tensor(sv)
                else:
                    self.optimizer.state[param][sk] = sv

    def save(self, filename='best_model.ckpt', epoch=0, val_metrics=None, intra_epoch_step=None, steps_per_epoch=None):
        path = os.path.join(self.checkpoint_dir, filename)

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        # 1. 用 EMA 权重保存模型参数 (和之前一样)
        with self.ema.average_parameters():
            ms.save_checkpoint(raw_model, path)

        # 2. 保存训练状态到伴随文件，用于断点续传
        # 注意：.ckpt 存的是 EMA 权重（用于推理），pkl 里额外存非 EMA 的训练权重（用于续传）
        state_path = path.replace('.ckpt', '_training_state.pkl')
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        training_state = {
            'epoch': epoch,
            'intra_epoch_step': intra_epoch_step,  # None 表示 epoch 结束，否则为 epoch 内的 step
            'global_step': self.global_step,
            'world_size': self._get_world_size(),
            'steps_per_epoch': steps_per_epoch,
            # 非 EMA 的真实训练权重（save 在 EMA context 外面，此时模型已恢复原始参数）
            'model_params': {p.name: p.data.asnumpy() for p in raw_model.get_parameters()},
            # optimizer state (用参数名作为 key，跨进程可匹配)
            'optimizer_state': self._serialize_optimizer_state(),
            'optimizer_param_groups_lr': [
                float(g['lr'].asnumpy()) if isinstance(g['lr'], (ms.Tensor, Parameter)) else float(g['lr'])
                for g in self.optimizer.param_groups
            ],
            # scheduler state
            'scheduler_last_epoch': self.scheduler.last_epoch if hasattr(self, 'scheduler') else None,
            # EMA shadow params
            'ema_shadow_params': [
                s.asnumpy() if s is not None else None
                for s in self.ema.shadow_params
            ],
            # val metrics for reference
            'val_metrics': val_metrics,
            # config flags
            'finetune_mode': self.finetune_mode,
        }

        with open(state_path, 'wb') as f:
            pickle.dump(training_state, f)

        print(f"✅ Checkpoint saved: {path} + {state_path}")

    def load_checkpoint(self, filename, new_steps_per_epoch=None):
        """从 checkpoint 恢复全部训练状态，用于断点续传

        Args:
            new_steps_per_epoch: 当前 world_size 下每 epoch 的步数，用于跨卡数续传时换算步数。
                                 如果为 None，则不做换算（假设卡数不变）。
        """
        path = os.path.join(self.checkpoint_dir, filename)
        state_path = path.replace('.ckpt', '_training_state.pkl')

        if not os.path.exists(state_path):
            print(f"⚠️ Training state file not found: {state_path}, only loading model weights.")
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            param_dict = ms.load_checkpoint(path)
            param_not_load, _ = ms.load_param_into_net(raw_model, param_dict)
            if param_not_load:
                print(f"⚠️ Parameters not loaded: {param_not_load}")
            return 0, None

        # 1. 加载训练状态
        with open(state_path, 'rb') as f:
            training_state = pickle.load(f)

        # 2. 加载模型权重：优先用 pkl 里的非 EMA 训练权重，保证续传正确
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        if 'model_params' in training_state:
            param_dict = {k: Parameter(ms.Tensor(v), name=k) for k, v in training_state['model_params'].items()}
            param_not_load, _ = ms.load_param_into_net(raw_model, param_dict)
            if param_not_load:
                print(f"⚠️ Parameters not loaded: {param_not_load}")
            print("✅ Loaded non-EMA training weights from pkl")
        else:
            # 兼容旧版 checkpoint：pkl 里没有 model_params，退回加载 .ckpt（EMA 权重）
            param_dict = ms.load_checkpoint(path)
            param_not_load, _ = ms.load_param_into_net(raw_model, param_dict)
            if param_not_load:
                print(f"⚠️ Parameters not loaded: {param_not_load}")
            print("⚠️ No non-EMA weights in pkl, loaded EMA weights from .ckpt as fallback")

        # --- 跨卡数续传：按比例换算步数 ---
        saved_global_step = training_state['global_step']
        saved_world_size = training_state.get('world_size', None)
        saved_steps_per_epoch = training_state.get('steps_per_epoch', None)
        cur_world_size = self._get_world_size()

        # 计算步数缩放比例：old_steps_per_epoch / new_steps_per_epoch
        step_scale = 1.0
        if (saved_world_size is not None and saved_world_size != cur_world_size
                and saved_steps_per_epoch is not None and new_steps_per_epoch is not None):
            step_scale = new_steps_per_epoch / saved_steps_per_epoch
            print(f"🔄 World size changed: {saved_world_size} → {cur_world_size}, "
                  f"steps_per_epoch: {saved_steps_per_epoch} → {new_steps_per_epoch}, "
                  f"step_scale: {step_scale:.4f}")

        self.global_step = int(saved_global_step * step_scale)
        if step_scale != 1.0:
            print(f"   global_step rescaled: {saved_global_step} → {self.global_step}")

        # 3. 恢复 optimizer state
        self._restore_optimizer_state(training_state.get('optimizer_state', {}))

        # 4. 恢复 scheduler state 并重新计算当前 lr
        # 注意：有 scheduler 时由 scheduler 控制 lr，无需手动恢复 optimizer lr
        if hasattr(self, 'scheduler') and training_state.get('scheduler_last_epoch') is not None:
            # 跨卡数续传时，scheduler 的 last_epoch 也要按比例换算
            saved_last_epoch = training_state['scheduler_last_epoch']
            rescaled_last_epoch = int(saved_last_epoch * step_scale)
            # 先设为 rescaled - 1，再调 step() 让 scheduler 自增回正确值
            # step() 会同时更新 optimizer.lrs 和 param_groups 的所有 lr 引用
            self.scheduler.last_epoch = rescaled_last_epoch - 1
            self.scheduler.step()
            if step_scale != 1.0:
                print(f"   scheduler.last_epoch rescaled: {saved_last_epoch} → {rescaled_last_epoch}")
        else:
            # 无 scheduler (finetune 模式): 手动恢复 optimizer lr
            saved_lrs = training_state.get('optimizer_param_groups_lr', [])
            for g, lr_val in zip(self.optimizer.param_groups, saved_lrs):
                if isinstance(g['lr'], (ms.Tensor, Parameter)):
                    g['lr'].set_data(ms.Tensor(lr_val, ms.float32))
                else:
                    g['lr'] = lr_val

        # 5. 恢复 EMA shadow params
        saved_ema = training_state.get('ema_shadow_params', [])
        for i, s in enumerate(saved_ema):
            if s is not None and i < len(self.ema.shadow_params):
                self.ema.shadow_params[i] = ms.Tensor(s)

        resumed_epoch = training_state['epoch']
        intra_step = training_state.get('intra_epoch_step', None)
        if intra_step is not None:
            # 跨卡数续传时，epoch 内步数也要按比例换算
            original_intra_step = intra_step
            intra_step = int(intra_step * step_scale)
            if step_scale != 1.0:
                print(f"   intra_epoch_step rescaled: {original_intra_step} → {intra_step}")
            print(f"✅ Resumed from epoch {resumed_epoch} step {intra_step}, global_step {self.global_step}")
        else:
            print(f"✅ Resumed from epoch {resumed_epoch} (completed), global_step {self.global_step}")
        return resumed_epoch, intra_step

