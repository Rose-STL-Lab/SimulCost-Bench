#!/bin/bash

set -e

LOG_FILE=".completed_runs.log"

# 如果日志文件不存在则创建
touch "$LOG_FILE"

MODELS=(
  "anthropic.claude-3-5-haiku-20241022-v1:0"
  "anthropic.claude-3-5-sonnet-20240620-v1:0"
  "anthropic.claude-3-7-sonnet-20250219-v1:0"
)

TASKS_1D=("cfl" "n_space")
TASKS_2D_ALL_MODES=("dx" "error_threshold")
TASKS_2D_ZERO_SHOT_ONLY=("relax" "t_init")

run_eval() {
  local MODEL="$1"
  local DIM="$2"
  local TASK="$3"
  local ZERO_SHOT="$4"

  local ID="model=$MODEL|dim=$DIM|task=$TASK|zero_shot=$ZERO_SHOT"

  if grep -Fxq "$ID" "$LOG_FILE"; then
    echo "Skipping completed: $ID"
    return
  fi

  echo "Running: $ID"
  if [[ "$ZERO_SHOT" == "true" ]]; then
    PYTHONPATH=$(pwd) python evaluation/heat_transfer/eval.py -m "$MODEL" -d "$DIM" -t "$TASK" -z
  else
    PYTHONPATH=$(pwd) python evaluation/heat_transfer/eval.py -m "$MODEL" -d "$DIM" -t "$TASK"
  fi

  echo "$ID" >> "$LOG_FILE"
}

echo "===== Running 1D_heat_transfer Experiments ====="
for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS_1D[@]}"; do
    run_eval "$MODEL" "1D_heat_transfer" "$TASK" true    # zero-shot
    run_eval "$MODEL" "1D_heat_transfer" "$TASK" false   # iterative
  done
done

echo "===== Running 2D_heat_transfer Experiments ====="
for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS_2D_ALL_MODES[@]}"; do
    run_eval "$MODEL" "2D_heat_transfer" "$TASK" true
    run_eval "$MODEL" "2D_heat_transfer" "$TASK" false
  done

  for TASK in "${TASKS_2D_ZERO_SHOT_ONLY[@]}"; do
    run_eval "$MODEL" "2D_heat_transfer" "$TASK" true
  done
done

echo "===== All Experiments Completed Successfully ====="
