#!/bin/bash
# model_inference_ns_transient_2d.sh
# Function: Execute NS Transient 2D model inference and evaluation in sequential order, stop on error; resume from failed command on next run
set -eE -o pipefail          # Exit immediately on any command or pipeline error, preserve ERR information

RESUME_LOG="scripts/inference_eval/ns_transient_2d_resume_progress.log"   # Log file for successful commands
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
tasks=("resolution" "cfl" "relaxation_factor" "residual_threshold")
precision_levels=("low" "medium" "high")
modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

model_provider="bedrock"
models=(
#  "anthropic.claude-3-5-haiku-20241022-v1:0"
#  "anthropic.claude-3-5-sonnet-20240620-v1:0"
 "anthropic.claude-3-7-sonnet-20250219-v1:0"
 "mistral.mistral-large-2402-v1:0"
 "meta.llama3-70b-instruct-v1:0"
 "amazon.nova-premier-v1:0"
)

# model_provider="custom_model"
# models=(
#  "qwen3_0_6b"
#  "qwen3_8b"
#  "qwen3_32b" 
#  "qwen3_235b_a22b"
# )

# model_provider="openai"
# models=(
#   "gpt-4o-mini"
# )

# ========= Main loop =========
for task in "${tasks[@]}"; do
  # Determine which modes to use for each task
  if [[ "$task" == "relaxation_factor" || "$task" == "residual_threshold" ]]; then
    # These tasks only support zero-shot mode
    task_modes=("-z")
  else
    # Other tasks support both modes
    task_modes=("${modes[@]}")
  fi
  
  for mode in "${task_modes[@]}"; do
    for precision_level in "${precision_levels[@]}"; do
      for model in "${models[@]}"; do
        # NS Transient 2D with precision level support
        run_cmd "python inference/langchain_LLM.py -p $model_provider -m $model -d ns_transient_2d -t $task -l $precision_level $mode --resume"
        run_cmd "python evaluation/ns_transient_2d/eval.py -m $model -d ns_transient_2d -t $task -l $precision_level $mode"
      done
    done
  done
done

echo "✅ All NS Transient 2D inference tasks completed successfully!"