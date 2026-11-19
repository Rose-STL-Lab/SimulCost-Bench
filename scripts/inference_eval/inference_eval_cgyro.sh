#!/bin/bash
# inference_eval_cgyro.sh
# Function: Execute CGYRO model inference and evaluation in sequential order, stop on error; resume from failed command on next run
set -eE -o pipefail

RESUME_LOG="scripts/inference_eval/cgyro_resume_progress.log"
touch "$RESUME_LOG"

# ========= Generic execution function =========
run_cmd () {
  local cmd="$1"
  if grep -Fxq "$cmd" "$RESUME_LOG"; then
    echo "✔ Already completed, skipping: $cmd"
    return
  fi
  echo "▶ Executing: $cmd"
  eval "$cmd"
  echo "$cmd" >> "$RESUME_LOG"
}

# ========= Parameter lists =========
tasks=("n_radial" "n_theta" "n_xi" "n_energy" "freq_tol" "delta_t")
precision_levels=("low" "medium" "high")
modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

model_provider="openai"
models=(
  "gpt-5-2025-08-07"
)

# ========= Main loop =========
for mode in "${modes[@]}"; do
  for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
      for precision_level in "${precision_levels[@]}"; do
        run_cmd "python inference/langchain_LLM.py -p $model_provider -m $model -d cgyro -t $task -l $precision_level $mode --resume"
        run_cmd "python evaluation/cgyro/eval.py -m $model -d cgyro -t $task -l $precision_level $mode"
      done
    done
  done
done

echo "✅ All CGYRO inference tasks completed successfully!"
