#!/bin/bash
# Script to launch Optuna study for SinoSheavesNN across 8 GPUs in a tmux session

SESSION_NAME="optuna_sinosheaves"
PROJECT_ROOT="/home/jdq/Master_Thesis"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
DB_PATH="$PROJECT_ROOT/db/optuna_snn_2d.db"
STUDY_NAME="snn_2d_optimization"
N_TRIALS_PER_GPU=50
N_EPOCHS=15
N_SAMPLES=100
BATCH_SIZE=2

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Killing existing tmux session '$SESSION_NAME'..."
    tmux kill-session -t $SESSION_NAME
fi

# Pre-create the study and SQLite tables sequentially to avoid Alembic migration race condition
echo "Initializing Optuna study '$STUDY_NAME' in SQLite..."
$PYTHON_BIN -c "
import optuna, sqlite3, sys
from pathlib import Path
db_path = Path('$DB_PATH')
db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_path), timeout=60)
conn.execute('PRAGMA journal_mode=WAL;')
conn.execute('PRAGMA busy_timeout=60000;')
conn.commit()
conn.close()

storage = optuna.storages.RDBStorage(url='sqlite:///$DB_PATH', engine_kwargs={'connect_args': {'timeout': 60}})
study = optuna.create_study(study_name='$STUDY_NAME', storage=storage, direction='maximize', load_if_exists=True)
print('Study initialized successfully.')
"

echo "Creating tmux session '$SESSION_NAME' with 8 GPU workers..."

# Window 0: GPU_0
tmux new-session -d -s $SESSION_NAME -n "GPU_0"
tmux send-keys -t $SESSION_NAME:GPU_0 "cd $PROJECT_ROOT" C-m
tmux send-keys -t $SESSION_NAME:GPU_0 "CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN src_2D/models/SinoSheavesNN/run_optuna.py --gpu-id 0 --n-trials $N_TRIALS_PER_GPU --n-epochs $N_EPOCHS --n-samples $N_SAMPLES --batch-size $BATCH_SIZE --use-wandb --storage sqlite:///$DB_PATH --study-name $STUDY_NAME" C-m

# Windows 1 to 7: GPU_1 to GPU_7
for GPU_ID in {1..7}; do
    sleep 0.5
    tmux new-window -t $SESSION_NAME -n "GPU_${GPU_ID}"
    tmux send-keys -t $SESSION_NAME:GPU_${GPU_ID} "cd $PROJECT_ROOT" C-m
    tmux send-keys -t $SESSION_NAME:GPU_${GPU_ID} "CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON_BIN src_2D/models/SinoSheavesNN/run_optuna.py --gpu-id 0 --n-trials $N_TRIALS_PER_GPU --n-epochs $N_EPOCHS --n-samples $N_SAMPLES --batch-size $BATCH_SIZE --use-wandb --storage sqlite:///$DB_PATH --study-name $STUDY_NAME" C-m
done

echo "Successfully launched Optuna study on 8 GPUs in tmux session '$SESSION_NAME'."
echo "To attach to the session, run: tmux attach -t $SESSION_NAME"
