# Script Layout

Scripts are command-line wrappers around the `src` library. They are not model
variants. Their names describe the action to run; architecture guarantees such
as SafeNorm and OMol/SPICE periodic-shift handling are documented in the root
README and implemented in `src`.

Inference scripts should pass dataset/head, charge, and spin into
`HTGP_Calculator`. They should not patch private graph-conversion methods.

Use these as the normal entry points:

- `training/finetune.py`: fine-tune a checkpoint on one H5 dataset.
- `training/launch_finetune_single.sh`: single-device fine-tune launcher.
- `training/launch_finetune_msrun.sh`: multi-device `msrun` fine-tune launcher.
- `inference/csp_relax.py`: production CSP relaxation/inference.
- `inference/ase_benchmark.py`: generic ASE single-point, optimization, MD, and speed benchmark.
- `data/prepare_h5_dataset.py`: convert extxyz ZIP data into H5 training format.
- `evaluation/validate_checkpoint.py`: validate a checkpoint on H5 datasets.

Advanced or debugging-only scripts:

- `training/train_multidataset.py`: hardcoded multi-dataset training/pretraining template.
- `diagnostics/singlepoint_probe.py`: single-point CSP evaluation without relaxation.
- `diagnostics/charge_spin_probe.py`: regression check that missing charge/spin behaves like explicit neutral charge/spin.
- `diagnostics/stress_smoke.py`: quick stress calculation smoke test.
