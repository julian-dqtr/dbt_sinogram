#!/bin/bash
# U-Net vs U-Net + HLCC with FEW and MANY training phantoms (docs/hlcc_projection.md, section 7).
# Four runs, ONE AFTER THE OTHER inside a single tmux session, each on the 8 GPUs (torchrun, DDP):
#
#   run name              model        training set                physics loss
#   UNet2D_N200           UNet2D       200 phantoms x 10 passes    none
#   UNet2dHLCC_N200       UNet2dHLCC   200 phantoms x 10 passes    HLCC orders 0-1
#   UNet2D                UNet2D       2000 phantoms               none
#   UNet2dHLCC            UNet2dHLCC   2000 phantoms               HLCC orders 0-1
#
# Everything else is the configuration of the UNet2D reference run (lr 3e-4, weight decay 1e-4,
# 32 filters, batch 4 per GPU, 200 epochs, annealing over 20 epochs) and every HLCC weight is LAMBDA.
# The small training set is repeated N_HIGH / N_LOW times per epoch, so that the four runs make exactly
# the same number of optimisation steps and validations, with the same learning-rate and annealing
# schedules: only the number of DISTINCT phantoms changes.
#
#   bash scripts/launch_unet_hlcc_data_study.sh                                    # the four runs
#   bash scripts/launch_unet_hlcc_data_study.sh UNet2D_N200 UNet2dHLCC_N200        # a subset, in this order
#   LAMBDA=3e-4 N_LOW=500 bash scripts/launch_unet_hlcc_data_study.sh              # other weight / small set
#   WANDB=0 bash scripts/launch_unet_hlcc_data_study.sh                            # without wandb
#   SEED=1 bash scripts/launch_unet_hlcc_data_study.sh                             # replicas: runs saved as <run>_S1
#
# All the runs of one seed share their initial weights and their training phantoms; another SEED
# changes the initialisation and the order of the batches (not the phantoms), which tells whether a
# difference between two runs is larger than the run-to-run noise.
#
# Follow it with `tmux attach -t unet_hlcc_data_study`; logs: outputs/2d/logs/unet_hlcc_data_study/<run>.log
# A finished run is marked by outputs/2d/checkpoints/<run>/DONE and skipped when the queue is relaunched.
# A run folder that already holds a checkpoint WITHOUT that marker (an older or interrupted run) is
# moved to outputs/backups/ first: a trained model is never overwritten.
# Afterwards:
#   .venv/bin/python src_2D/evaluate_all.py --models UNet2D UNet2dHLCC UNet2D_N200 UNet2dHLCC_N200

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
TORCHRUN_BIN="$PROJECT_ROOT/.venv/bin/torchrun"
SESSION_NAME="unet_hlcc_data_study"
N_GPUS=8
EPOCHS="${EPOCHS:-200}"
N_HIGH="${N_HIGH:-2000}"
N_LOW="${N_LOW:-200}"
LAMBDA="${LAMBDA:-1e-3}"
WANDB="${WANDB:-1}"
SEED="${SEED:-0}"
BACKUP_STAMP="${BACKUP_STAMP:-$(date '+%Y-%m-%d_%H%M%S')}"

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=("UNet2D_N${N_LOW}" "UNet2dHLCC_N${N_LOW}" UNet2D UNet2dHLCC)
for RUN in "${RUNS[@]}"; do
    if ! [[ "$RUN" =~ ^(UNet2D|UNet2dHLCC)(_N[0-9]+)?$ ]]; then
        echo "Invalid run '$RUN': expected UNet2D[_N<n>] or UNet2dHLCC[_N<n>], e.g. UNet2dHLCC_N200" >&2
        exit 2
    fi
    if [[ "$RUN" =~ _N([0-9]+)$ ]] && [ $(( N_HIGH % BASH_REMATCH[1] )) -ne 0 ]; then
        echo "Invalid run '$RUN': the small training set must divide N_HIGH = $N_HIGH" >&2
        exit 2
    fi
done

# Outside tmux: start the queue in a detached session (it survives the end of the SSH session).
if [ "${DATA_STUDY_INSIDE_TMUX:-0}" != "1" ]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME" >&2
        echo "Kill it first if you really want to restart the queue: tmux kill-session -t $SESSION_NAME" >&2
        exit 1
    fi
    tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && DATA_STUDY_INSIDE_TMUX=1 EPOCHS='$EPOCHS' \
N_HIGH='$N_HIGH' N_LOW='$N_LOW' LAMBDA='$LAMBDA' WANDB='$WANDB' SEED='$SEED' BACKUP_STAMP='$BACKUP_STAMP' bash '$SCRIPT_PATH' ${RUNS[*]}; read"
    echo "Queue started in tmux session '$SESSION_NAME': ${RUNS[*]}"
    echo "Attach with: tmux attach -t $SESSION_NAME   (logs: outputs/2d/logs/$SESSION_NAME/)"
    exit 0
fi

log() { echo "[$(date '+%F %T')] $*"; }

cd "$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/outputs/2d/logs/$SESSION_NAME"
mkdir -p "$LOG_DIR"
FAILED=()

run_training() {
    local spec="$1"
    local model="${spec%%_*}"
    local run="$spec"
    [ "$SEED" != "0" ] && run="${spec}_S${SEED}"
    local checkpoint_dir="$PROJECT_ROOT/outputs/2d/checkpoints/$run"
    if [ -f "$checkpoint_dir/DONE" ]; then
        log "$run: already done, skipped"
        return 0
    fi
    if [ -f "$checkpoint_dir/best_model.pt" ]; then
        local backup_dir="$PROJECT_ROOT/outputs/backups/checkpoints_$BACKUP_STAMP"
        mkdir -p "$backup_dir"
        mv "$checkpoint_dir" "$backup_dir/$run"
        log "$run: previous checkpoint moved to outputs/backups/checkpoints_$BACKUP_STAMP/$run"
    fi

    local n_samples="$N_HIGH" repeats=1 extra=()
    if [[ "$spec" =~ _N([0-9]+)$ ]]; then
        n_samples="${BASH_REMATCH[1]}"
        repeats=$(( N_HIGH / n_samples ))
    fi
    if [ "$model" = "UNet2dHLCC" ]; then
        extra+=(--lambda_m0 "$LAMBDA" --lambda_m1 "$LAMBDA" --anneal_epochs 20)
    fi
    [ "$WANDB" = "1" ] && extra+=(--use-wandb)

    log "$run: $model, $n_samples phantoms x $repeats pass(es) per epoch, $EPOCHS epochs, $N_GPUS GPUs (log: $LOG_DIR/$run.log)"
    if "$TORCHRUN_BIN" --nproc_per_node="$N_GPUS" src_2D/train.py --model "$model" --run_name "$run" \
            --epochs "$EPOCHS" --n_samples "$n_samples" --train_repeats "$repeats" --n_val 200 --batch_size 4 \
            --lr 3e-4 --weight_decay 1e-4 --filters 32 --seed "$SEED" "${extra[@]}" > "$LOG_DIR/$run.log" 2>&1 \
            && [ -f "$checkpoint_dir/training_stats.json" ]; then
        touch "$checkpoint_dir/DONE"
        log "$run: done. $(grep -m1 '^Done\.' "$LOG_DIR/$run.log" || true)"
    else
        FAILED+=("$run")
        log "$run: FAILED, see $LOG_DIR/$run.log"
    fi
}

log "U-Net / HLCC data study: ${RUNS[*]} (lambda = $LAMBDA, small training set = $N_LOW phantoms, seed = $SEED)"
for RUN in "${RUNS[@]}"; do
    run_training "$RUN"
done
if [ ${#FAILED[@]} -eq 0 ]; then
    SUFFIX=""
    [ "$SEED" != "0" ] && SUFFIX="_S${SEED}"
    log "Queue finished. Evaluate with: .venv/bin/python src_2D/evaluate_all.py --models ${RUNS[*]/%/$SUFFIX}"
else
    log "Queue finished with ${#FAILED[@]} failed run(s): ${FAILED[*]}"
fi
