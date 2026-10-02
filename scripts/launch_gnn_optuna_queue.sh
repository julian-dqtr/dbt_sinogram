#!/bin/bash
# Phase 1 of docs/plan_experiences_gnn.md: the six small Optuna studies of the graph models
# (GCN and SNN at 6, 12 and 18 layers), run ONE AFTER THE OTHER inside a single tmux session.
# Each study uses the 8 GPUs (one worker per GPU) and searches lr and weight_decay only
# (--optimizer-only): k = 12, sigma_deg = 5 and num_stalks = 32 keep their train.py defaults, so
# GCN and SNN share the same graph and the same parameter count at every depth.
#
#   bash scripts/launch_gnn_optuna_queue.sh                         # GCN:6 SNN:6 GCN:12 SNN:12 GCN:18 SNN:18
#   bash scripts/launch_gnn_optuna_queue.sh GCN:18 SNN:18           # a subset, in this order
#   N_TRIALS_PER_GPU=6 bash scripts/launch_gnn_optuna_queue.sh      # 48 trials per study instead of 32
#   EXTRA_ARGS="--grad_checkpoint" bash scripts/launch_gnn_optuna_queue.sh GCN:18 SNN:18   # if L18 runs out of memory
#
# Follow it with `tmux attach -t optuna_gnn_queue`; worker logs: outputs/2d/optuna/<study>/logs/.
# A finished study is marked by outputs/2d/optuna/<study>/DONE and skipped when the queue is
# relaunched, so an interrupted queue resumes at the first unfinished study. The queue stops if
# every worker of a study fails (e.g. out of GPU memory) instead of wasting the next studies.
# Final trainings, afterwards: .venv/bin/python scripts/launch_best_training.py <study> --run

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
SESSION_NAME="optuna_gnn_queue"
N_GPUS=8
N_TRIALS_PER_GPU="${N_TRIALS_PER_GPU:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Same trial budget as scripts/launch_optuna_8gpus.sh
N_EPOCHS=30
N_SAMPLES=500
N_VAL=100
BATCH_SIZE=4

SPECS=("$@")
[ ${#SPECS[@]} -eq 0 ] && SPECS=(GCN:6 SNN:6 GCN:12 SNN:12 GCN:18 SNN:18)
for SPEC in "${SPECS[@]}"; do
    if ! [[ "$SPEC" =~ ^(GCN|SNN):[0-9]+$ ]]; then
        echo "Invalid study '$SPEC': expected <GCN|SNN>:<num_layers>, e.g. SNN:12" >&2
        exit 2
    fi
done

# Outside tmux: start the queue in a detached session (it survives the end of the SSH session).
if [ "${OPTUNA_QUEUE_INSIDE_TMUX:-0}" != "1" ]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME" >&2
        echo "Kill it first if you really want to restart the queue: tmux kill-session -t $SESSION_NAME" >&2
        exit 1
    fi
    tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && OPTUNA_QUEUE_INSIDE_TMUX=1 \
N_TRIALS_PER_GPU='$N_TRIALS_PER_GPU' EXTRA_ARGS='$EXTRA_ARGS' bash '$SCRIPT_PATH' ${SPECS[*]}; read"
    echo "Queue started in tmux session '$SESSION_NAME': ${SPECS[*]} ($N_GPUS workers x $N_TRIALS_PER_GPU trials per study)"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

log() { echo "[$(date '+%F %T')] $*"; }

run_study() {
    local model="$1" num_layers="$2"
    local study="${model}_L${num_layers}_parallel_optimizer_only"
    local out_dir="$PROJECT_ROOT/outputs/2d/optuna/$study"
    if [ -f "$out_dir/DONE" ]; then
        log "$study: already done, skipped"
        return 0
    fi
    mkdir -p "$out_dir/logs"
    local common=(src_2D/optuna_search.py --model "$model" --num_layers "$num_layers" --optimizer-only
                  --n-epochs "$N_EPOCHS" --n-samples "$N_SAMPLES" --n-val "$N_VAL" --batch-size "$BATCH_SIZE"
                  --study-name "$study" --storage "sqlite:///$PROJECT_ROOT/db/optuna_${study}.db")
    # Create the study once, before the workers, to avoid a table-creation race between them.
    "$PYTHON_BIN" "${common[@]}" --create-only

    log "$study: $N_GPUS workers x $N_TRIALS_PER_GPU trials (logs: $out_dir/logs/)"
    local pids=() gpu pid failed=0
    for gpu in $(seq 0 $((N_GPUS - 1))); do
        # shellcheck disable=SC2086  # EXTRA_ARGS is a list of flags
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${common[@]}" --n-trials "$N_TRIALS_PER_GPU" --use-wandb $EXTRA_ARGS \
            > "$out_dir/logs/gpu${gpu}.log" 2>&1 &
        pids+=("$!")
        sleep 0.5
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=$((failed + 1))
    done

    if [ "$failed" -eq "$N_GPUS" ]; then
        log "$study: all $N_GPUS workers failed, queue stopped. See $out_dir/logs/"
        return 1
    fi
    touch "$out_dir/DONE"
    log "$study: done, $failed failed worker(s). $(grep -m1 '"best_' "$out_dir/best_params.json" 2>/dev/null || echo 'no best_params.json')"
}

log "Optuna queue: ${SPECS[*]}"
for SPEC in "${SPECS[@]}"; do
    run_study "${SPEC%%:*}" "${SPEC##*:}"
done
log "Queue finished. Final trainings: .venv/bin/python scripts/launch_best_training.py <study> --run"
