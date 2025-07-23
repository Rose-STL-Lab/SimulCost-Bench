#!/bin/bash
# model_inference_heat_1d.sh
# Function: Execute 1D Heat Transfer model inference and evaluation in sequential order, stop on error; resume from failed command on next run
set -eE -o pipefail          # Exit immediately on any command or pipeline error, preserve ERR information

RESUME_LOG="scripts/inference_eval/heat_1d_resume_progress.log"   # Log file for successful commands
touch "$RESUME_LOG"

# ========= Generic execution function =========
run_cmd () {
  local cmd="$1"
  # Skip if command is already logged
  if grep -Fxq "$cmd" "$RESUME_LOG"; then
    echo "✔ Already completed, skipping: $cmd"
    return
  fi

  echo "▶ Executing: $cmd"
  eval "$cmd"               # Run actual command

  # Only append to log if successful; failure triggers set -eE to exit directly, no log written
  echo "$cmd" >> "$RESUME_LOG"
}

# ========= Parameter lists =========
tasks=("cfl" "n_space")
modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

model_provider="custom_model"
models=(
  "qwen3_8b"
)

# ========= Main loop =========
for mode in "${modes[@]}"; do
  for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
      # 1D Heat Transfer has no case concept, run tasks directly
      run_cmd "python inference/langchain_LLM.py -p $model_provider -m $model -d 1D_heat_transfer -t $task $mode"
      run_cmd "python evaluation/heat_transfer/eval.py -m $model -d 1D_heat_transfer -t $task $mode"
    done
  done
done

echo "✅ All 1D Heat Transfer inference tasks completed successfully!"