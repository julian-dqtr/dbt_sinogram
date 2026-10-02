# Limited-angle sinogram completion (Master's thesis)

Parallel-beam sinogram completion (180 views over [-90, 90) degrees, acquired window +/-25
degrees) with four learned models sharing ONE protocol:

| Name | Model | Loss |
|---|---|---|
| `UNet2D` | U-Net (residual, soft data consistency) | MSE |
| `UNet2dHLCC` | the same network | MSE + annealed Helgason-Ludwig (HLCC) penalty |
| `GCN` | view-graph GCN, kNN k = 12, no rotation | MSE (`--physics hlcc` optional) |
| `SNN` | SinoSheavesNN: same graph and parameters, hard-coded SO(2) maps R(delta theta) | idem |

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

# Evaluation on the immutable test split (+ per-view error profile)
.venv/bin/python src_2D/evaluate_all.py
```

## Protocol (what every number in the thesis relies on)

- **Geometry**: `src_2D/conf/geometry_conf_2d.py` (`beam="parallel"`). The ground truth
  satisfies the HLCC (`tests/test_physics_loss.py`); the legacy stationary-detector fan-beam
  (`beam="fanflat_vec"`) does not and is kept only to document that fact.
- **Data**: phantoms generated on the fly from per-sample seeds. `train`, `val` and `test`
  are fixed, disjoint and independent of the global RNG. Inputs are noisy (Poisson), targets
  are clean.
- **Data consistency**: soft mask tapered INSIDE the acquired window, identical for all
  learned models, applied inside `model(incomplete)`.
- **Metrics**: `src_2D/utils/metrics.py`, fixed `data_range = 1.0`, reported on the whole
  sinogram and on the missing wedge. Checkpoints are selected on the validation wedge MSE.
- **Checkpoints** are self-describing (architecture + geometry + metrics + git commit);
  `src_2D/models/factory.get_model` loads them strictly and raises on any mismatch.

Theory notes: `docs/harmonic_sheaves.md` (numbers reproduced by `scripts/verify_harmonic_sheaves.py`).
`src_2D/models/GLM` and `src_2D/models/Unet2dRNO` are legacy fan-beam experiments outside this protocol.
