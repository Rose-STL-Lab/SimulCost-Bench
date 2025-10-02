#!/bin/bash
# model_inference_heat_1d_bo.sh
# Function: Execute Heat 1D BO model inference and evaluation in sequential order, stop on error; resume from failed command on next run
set -eE -o pipefail          # Exit immediately on any command or pipeline error, preserve ERR information

RESUME_LOG="scripts/inference_eval/heat_1d_bo_resume_progress.log"   # Log file for successful commands
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
precision_levels=("low" "medium" "high")
modes=("-z")

# Bayesian Optimization specific models
model_provider="custom_model"
models=(
 "bayesian_optimization"
)

# ========= Main loop =========
for mode in "${modes[@]}"; do
  for task in "${tasks[@]}"; do
    for precision_level in "${precision_levels[@]}"; do
      for model in "${models[@]}"; do
        # Heat 1D BO with precision level support and profile passing
        run_cmd "python inference/langchain_LLM.py -p $model_provider -m $model -d heat_1d_bo -t $task -l $precision_level $mode --resume"
        # Use heat_1d evaluation since heat_1d_bo shares the same evaluation logic
        run_cmd "python evaluation/heat_1d/eval.py -m $model -d heat_1d_bo -t $task -l $precision_level $mode"
      done
    done
  done
done

echo "✅ All Heat 1D BO inference tasks completed successfully!"
echo "📊 Results can be found in:"
echo "   - results_model_attempt/heat_1d_bo/"
echo "   - log_model_tool_call/heat_1d_bo/"
echo ""
echo "🔍 To compare with regular heat_1d results, check:"
echo "   - results_model_attempt/heat_1d/"
echo ""
echo "💡 This script tested profile passing functionality for Bayesian Optimization on heat_1d_bo dataset."