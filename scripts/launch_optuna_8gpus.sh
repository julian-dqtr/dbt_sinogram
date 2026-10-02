#!/bin/bash
# Launch one Optuna worker per GPU (8 GPUs) in a tmux session, all sharing the same study.
#
#   bash scripts/launch_optuna_8gpus.sh UNet2D
#   bash scripts/launch_optuna_8gpus.sh UNet2dHLCC
#   bash scripts/launch_optuna_8gpus.sh GCN 12          # model, num_layers
#   bash scripts/launch_optuna_8gpus.sh SNN 18 hlcc     # model, num_layers, physics (auto|none|hlcc)
#
# Each worker only sees its own GPU through CUDA_VISIBLE_DEVICES (no --gpu-id: combining
# both made 7 workers out of 8 crash on an invalid device index).

set -euo pipefail

MODEL="${1:?Usage: $0 <UNet2D|UNet2dHLCC|GCN|SNN> [num_layers] [physics]}"
NUM_LAYERS="${2:-6}"
PHYSICS="${3:-auto}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
N_GPUS=8
N_TRIALS_PER_GPU=25
N_EPOCHS=30
N_SAMPLES=500
N_VAL=100
BATCH_SIZE=4

case "$MODEL" in
    GCN|SNN) RUN_NAME="${MODEL}_L${NUM_LAYERS}" ;;
    *)       RUN_NAME="$MODEL" ;;
esac
STUDY_NAME="${RUN_NAME}_parallel"
[ "$PHYSICS" != "auto" ] && STUDY_NAME="${STUDY_NAME}_${PHYSICS}"
SESSION_NAME="optuna_${STUDY_NAME}"
DB_PATH="$PROJECT_ROOT/db/optuna_${STUDY_NAME}.db"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
    echo "Kill it first if you really want to restart the workers: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create the study sequentially to avoid a table-creation race between the workers.
"$PYTHON_BIN" - <<PY
import optuna, sqlite3
from pathlib import Path
db_path = Path("$DB_PATH"); db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_path), timeout=60)
conn.execute("PRAGMA journal_mode=WAL;"); conn.commit(); conn.close()
storage = optuna.storages.RDBStorage(url="sqlite:///$DB_PATH", engine_kwargs={"connect_args": {"timeout": 60}})
optuna.create_study(study_name="$STUDY_NAME", storage=storage, direction="minimize", load_if_exists=True)
print("Study '$STUDY_NAME' ready.")
PY

CMD="$PYTHON_BIN src_2D/optuna_search.py --model $MODEL --num_layers $NUM_LAYERS --physics $PHYSICS \
--n-trials $N_TRIALS_PER_GPU --n-epochs $N_EPOCHS --n-samples $N_SAMPLES --n-val $N_VAL --batch-size $BATCH_SIZE \
--study-name $STUDY_NAME --storage sqlite:///$DB_PATH --use-wandb"

tmux new-session -d -s "$SESSION_NAME" -n "GPU_0" "cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=0 $CMD; read"
for GPU_ID in $(seq 1 $((N_GPUS - 1))); do
    sleep 0.5
    tmux new-window -t "$SESSION_NAME" -n "GPU_${GPU_ID}" "cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=${GPU_ID} $CMD; read"
done

echo "Launched $N_GPUS workers on study '$STUDY_NAME'. Attach with: tmux attach -t $SESSION_NAME"
echo "Best parameters are rewritten after every trial in outputs/2d/optuna/$STUDY_NAME/best_params.json"
