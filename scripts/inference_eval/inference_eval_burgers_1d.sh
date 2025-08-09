#!/bin/bash
# run_all_resume.sh
# Function: Execute all commands in sequential order, stop on error; resume from failed command on next run
set -eE -o pipefail          # Exit immediately on any command or pipeline error, preserve ERR information

RESUME_LOG="scripts/inference_eval/burgers_1d_resume_progress.log"   # Log file for successful commands
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
tasks=("cfl" "k" "w")
cases=("blast" "double_shock" "rarefaction" "sin" "sod")
modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

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
for mode in "${modes[@]}"; do
  for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
      for case in "${cases[@]}"; do
        run_cmd "python inference/langchain_LLM.py -p $model_provider -m $model -d burgers_1d -t $task -c $case $mode --resume"
        run_cmd "python evaluation/burgers/eval.py -m $model -d burgers_1d -t $task -c $case $mode"
      done
    done
  done
done

echo "✅ All tasks completed successfully!"
