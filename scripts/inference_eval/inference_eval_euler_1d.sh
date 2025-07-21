#!/bin/bash
# model_inference_euler_1d.sh
# Function: Execute 1D Euler equation model inference and evaluation in sequential order, stop on error; resume from failed command on next run
set -eE -o pipefail          # Exit immediately on any command or pipeline error, preserve ERR information

RESUME_LOG="scripts/inference_eval/euler_1d_resume_progress.log"   # Log file for successful commands
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
tasks=("cfl" "beta" "k")
models=(
  "qwen3_0_6b"
)
cases=("sod")  # 1D Euler only has sod test case
modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

# ========= Main loop =========
for mode in "${modes[@]}"; do
  for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
      for case in "${cases[@]}"; do
        run_cmd "python inference/langchain_LLM.py -p custom_model -m $model -d euler_1d -t $task -c $case $mode"
        run_cmd "PYTHONPATH=$(pwd) python evaluation/euler/eval.py -m $model -d euler_1d -t $task -c $case $mode"
      done
    done
  done
done

echo "✅ All 1D Euler equation inference tasks completed successfully!"