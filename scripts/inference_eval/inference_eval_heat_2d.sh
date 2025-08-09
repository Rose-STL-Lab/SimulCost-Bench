#!/bin/bash
# model_inference_heat_2d.sh
# Function: Execute 2D Heat Transfer model inference and evaluation in sequential order, stop on error; resume from failed command on next run
set -eE -o pipefail          # Exit immediately on any command or pipeline error, preserve ERR information

RESUME_LOG="scripts/inference_eval/heat_2d_resume_progress.log"   # Log file for successful commands
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
# Define tasks and their corresponding modes
declare -A task_modes
task_modes["dx"]="-z "           # dx supports zero-shot and iterative
task_modes["error_threshold"]="-z "  # error_threshold supports zero-shot and iterative  
task_modes["relax"]="-z"         # relax only supports zero-shot
task_modes["t_init"]="-z"        # t_init only supports zero-shot

#model_provider="bedrock"
#models=(
#  "anthropic.claude-3-5-haiku-20241022-v1:0"
#  "anthropic.claude-3-5-sonnet-20240620-v1:0"
#  "anthropic.claude-3-7-sonnet-20250219-v1:0"
#  "mistral.mistral-large-2402-v1:0"
#  "meta.llama3-70b-instruct-v1:0"
#  "amazon.nova-premier-v1:0"
#)

model_provider="openai"
models=(
  "gpt-4o-mini"
)
# ========= Main loop =========
for task in "${!task_modes[@]}"; do
  modes_str="${task_modes[$task]}"
  
  if [[ "$modes_str" == *" "* ]]; then
    # Contains space, supports both modes
    modes=("-z" "")
  else
    # Only supports one mode
    modes=("$modes_str")
  fi
  
  for mode in "${modes[@]}"; do
    for model in "${models[@]}"; do
      # 2D Heat Transfer has no case concept, run tasks directly
      run_cmd "python inference/langchain_LLM.py -n 100 -p $model_provider -m $model -d 2D_heat_transfer -t $task $mode --resume"
      run_cmd "python evaluation/heat_transfer/eval.py -m $model -d 2D_heat_transfer -t $task $mode"
    done
  done
done

echo "✅ All 2D Heat Transfer inference tasks completed successfully!"