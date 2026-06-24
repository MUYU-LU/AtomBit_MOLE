# AtomBit Official Clean Package

Clean AtomBit package with one source tree only:

- `src/`: unified training and inference source.
- `scripts/training/`: training and fine-tuning entry points.
- `scripts/inference/`: production inference and ASE benchmark entry points.
- `scripts/data/`: dataset conversion utilities.
- `scripts/evaluation/`: checkpoint evaluation on H5 datasets.
- `scripts/diagnostics/`: debugging probes and smoke tests.
- `sharker/`: bundled runtime dependency used by the loaders.
- `e0/`: packaged original E0 baselines.

`sharker` is a required local runtime dependency from the original AtomBit
codebase. It provides graph containers, batching, and dataloader utilities used
by `src`; it is not a model variant and not an inference script.

Scripts are intentionally named by task, not by model variant. The SafeNorm and
OMol/SPICE choices are package-level guarantees documented below and implemented
in `src`.

## What This Package Guarantees

- L1/L2 equivariant channels use `SafeEquivariantRMSNorm(..., min_rms=0.05)`.
- Raw/original L1/L2 `RMSNorm` source variants are not included.
- OMol/SPICE periodic inference keeps periodic neighbor shifts for periodic CSP crystals.
- Training and inference both import the same package path: `src`.
- Bare ASE structures default to neutral `charge=0`, `spin=0`; neutral charge maps to embedding index `0 - min_charge`, not row `0`.
- Inference dataset/head, charge, spin, periodic shifts, and E0 lookup are handled by `HTGP_Calculator`; scripts are only thin wrappers.

## Training

User fine-tuning:

```bash
python scripts/training/finetune.py \
  --ckpt /path/to/base.ckpt \
  --data_dir /path/to/train_data \
  --dataset_name mpa \
  --epochs 1 \
  --lr 1e-4 \
  --batch_cost 1000 \
  --output_dir ./finetune_out \
  --progress_file ./finetune_out/progress.jsonl
```

Multi-device fine-tuning is launched externally with `msrun`; see
`scripts/training/launch_finetune_msrun.sh`.

Full multi-dataset training/pretraining template:

```bash
python scripts/training/train_multidataset.py
```

Before using `train_multidataset.py`, edit its `Config.DATA_DIR`, `DATA_DIR_NAMES`,
checkpoint, epoch, and batch-cost settings.

## Data Conversion

Convert extxyz ZIP data into the H5/metadata format used by `finetune.py`:

```bash
python scripts/data/prepare_h5_dataset.py \
  --zip /path/to/data.zip \
  --output_dir /path/to/train_data \
  --dataset_context OMat24 \
  --cutoff 6.0
```

## Inference

Recommended CSP relaxation for neutral CIFs:

```bash
python scripts/inference/csp_relax.py \
  --work-dir /path/to/csp_workdir \
  --checkpoint /path/to/model.ckpt \
  --dataset-name OMol25 \
  --molecule-single 1 \
  --n-jobs 1 \
  --smoke 4
```

The work directory must contain `structures/*.cif`. The script passes
`--dataset-name` into `HTGP_Calculator`; the calculator assigns neutral
`charge=0`, `spin=0` unless `atoms.info` overrides them, and defaults to this
package's `src` and `e0/` directories.

Generic ASE benchmark, single point, optimization, and MD:

```bash
python scripts/inference/ase_benchmark.py \
  --ckpt /path/to/model.ckpt \
  --device Ascend \
  --dataset OMol25 \
  --structure /path/to/structure.cif
```

Single-point matrix/probe:

```bash
python scripts/diagnostics/singlepoint_probe.py \
  --structures-dir /path/to/structures \
  --json-dir /path/to/json_result \
  --lock-dir /path/to/locks \
  --checkpoint /path/to/model.ckpt \
  --checkpoint-label safe \
  --norm-label SafeEquivariantRMSNorm \
  --dataset-name OMol25
```

## Run Metadata To Record

For every formal run, record:

- package path: `AtomBit_Official_Clean`
- checkpoint path
- E0 source: packaged `e0/` or user-fitted E0 JSON
- dataset/head name
- charge/spin handling
- whether periodic CSP shifts were preserved
- stress validation result if stress is reported
