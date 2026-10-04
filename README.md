# Limited-angle sinogram completion (Master's thesis)

Parallel-beam sinogram completion (180 views over [-90, 90) degrees, acquired window +/-25
degrees) with four learned models sharing ONE protocol:

| Name | Model | Loss |
|---|---|---|
| `UNet2D` | U-Net (residual, hard data consistency) | MSE |
| `UNet2dHLCC` | the same network | MSE + annealed Helgason-Ludwig (HLCC) penalty, orders 0-1 (`--hlcc_max_order 3`: orders 0-3) |
| `GCN` | view-graph GCN, kNN k = 12, identity transport | MSE (`--physics hlcc` optional) |
| `SNN` | SinoSheavesNN: same graph and parameters, hard-coded shift maps: stalk f translated along the detector by t_f (theta_i - theta_j), one depth t_f per stalk | idem |

Graph models are compared at depths 6 / 12 / 18 (`--num_layers`): their angular reach is
`num_layers * k/2` views = 36 / 72 / 108 degrees, while the farthest missing view is 65
degrees away from the acquired window.

## Commands

```bash
# Tests (geometry, HLCC loss on the ground truth, DC mask, seeded splits, models, end-to-end)
.venv/bin/python -m pytest tests

# Training (single GPU, or torchrun for DDP). Checkpoints: outputs/2d/checkpoints/<run_name>/
.venv/bin/python src_2D/train.py --model UNet2D --use-wandb
.venv/bin/python src_2D/train.py --model UNet2dHLCC --use-wandb
.venv/bin/torchrun --nproc_per_node=8 src_2D/train.py --model GCN --num_layers 12 --use-wandb
.venv/bin/torchrun --nproc_per_node=8 src_2D/train.py --model SNN --num_layers 18 --use-wandb

# Optuna: one worker per GPU in tmux, then the training command of the best trial
bash scripts/launch_optuna_8gpus.sh SNN 12
.venv/bin/python scripts/launch_best_training.py SNN_L12_parallel            # prints the command (--run to start it)

# Graph models (docs/plan_experiences_gnn.md): the six lr / weight_decay studies, one after the other
bash scripts/launch_gnn_optuna_queue.sh
.venv/bin/python scripts/launch_best_training.py GCN_L6_parallel_optimizer_only --run
.venv/bin/python scripts/launch_best_training.py GCN_L12_parallel_optimizer_only --model SNN --run   # SNN, GCN configuration

# Evaluation on the immutable test split (+ per-view error profile and inter-sample variance)
.venv/bin/python src_2D/evaluate_all.py
# Analysis: run results/model_analysis.ipynb. It writes the thesis tables (CSV + LaTeX, booktabs) to results/tables/
# and the figures (PDF) to results/figures/, both versioned. Peak VRAM of the runs trained before 2026-10-04:
.venv/bin/python scripts/measure_peak_vram.py

# U-Net vs U-Net + HLCC with few (200) and many (2000) training phantoms, one run after the other,
# then the paired comparison on the test split (tables + figures)
bash scripts/launch_unet_hlcc_data_study.sh
.venv/bin/python src_2D/evaluate_all.py --models $(.venv/bin/python scripts/data_study_report.py --print_models)
.venv/bin/python scripts/data_study_report.py

# Data consistency: the acquired views of the outputs minus the measured ones (must print exactly 0)
.venv/bin/python scripts/check_data_consistency.py --models UNet2D UNet2dHLCC

# HLCC moment regression (Huang et al. 2017): why it stops at order 4 (figure + table, inference only)
.venv/bin/python scripts/hlcc_order_study.py
```

Model names accept suffixes (`src_2D/models/factory.py`): `GCN_L12` (depth), `UNet2D_N200`
(the run saved under that name with `train.py --run_name`), and `UNet2D_P4` (the
model followed by the HLCC moment regression of orders 0..4 on the missing views, no training;
`UNet2D_P3` stops at order 3).

## Protocol (what every number in the thesis relies on)

- **Geometry**: `src_2D/conf/geometry_conf_2d.py` (`beam="parallel"`). The ground truth
  satisfies the HLCC (`tests/test_physics_loss.py`); the legacy stationary-detector fan-beam
  (`beam="fanflat_vec"`) does not and is kept only to document that fact.
- **Data**: phantoms generated on the fly from per-sample seeds. `train`, `val` and `test`
  are fixed, disjoint and independent of the global RNG. Inputs are noisy (Poisson), targets
  are clean.
- **Data consistency**: hard, identical for all learned models, applied inside
  `model(incomplete)`: every acquired view comes out bit for bit, the missing views are never
  touched. (The runs trained before 2026-09-30 used a soft mask tapered inside the window; it made
  the 8 border views 15 to 19 times less accurate than the measurements, see `docs/hlcc_projection.md`.)
- **HLCC**: `src_2D/utils/hlcc.py`. Penalty of orders 0..`--hlcc_max_order` in the loss of
  `UNet2dHLCC` (default weights 1e-3), residuals `hlcc_residual_<n>` in the evaluation, and the
  optional moment regression `_P4` as a post-processing.
- **Metrics**: `src_2D/utils/metrics.py`, fixed `data_range = 1.0`, reported on the whole
  sinogram and on the missing wedge. Checkpoints are selected on the validation wedge MSE.
- **Checkpoints** are self-describing (architecture + geometry + metrics + git commit);
  `src_2D/models/factory.get_model` loads them strictly and raises on any mismatch.

Theory notes: `docs/harmonic_sheaves.md` (numbers reproduced by `scripts/verify_harmonic_sheaves.py`)
and `docs/hlcc_projection.md` (HLCC loss, moment regression, hard data consistency, data regimes).
`src_2D/models/GLM` is a legacy fan-beam experiment outside this protocol.
