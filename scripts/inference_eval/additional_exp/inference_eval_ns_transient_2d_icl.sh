#!/bin/bash
# inference_eval_ns_transient_2d_icl.sh
# Function: Execute NS Transient 2D ICL variants model inference and evaluation in sequential order
# Runs ns_transient_2d_icl_accuracy_focused, ns_transient_2d_icl_cost_excluded, ns_transient_2d_icl_full datasets
# Stops on error; resumes from failed command on next run
set -eE -o pipefail          # Exit immediately on any command or pipeline error, preserve ERR information

RESUME_LOG="scripts/inference_eval/additional_exp/ns_transient_2d_icl_resume_progress.log"   # Log file for successful commands
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
icl_datasets=("ns_transient_2d_icl_accuracy_focused" "ns_transient_2d_icl_cost_excluded" "ns_transient_2d_icl_full")
tasks=("resolution" "cfl" "relaxation_factor" "residual_threshold")
precision_levels=("low" "medium" "high")
# modes=("-z" "")   # "-z" for zero-shot, empty string for iterative
modes=("")

# ========= Model configuration =========
# Configure one provider and corresponding models at a time

# Bedrock models
model_provider="bedrock"
models=(
  "anthropic.claude-3-7-sonnet-20250219-v1:0"
  # "mistral.mistral-large-2402-v1:0"
  # "meta.llama3-70b-instruct-v1:0"
  # "amazon.nova-premier-v1:0"
)

# Custom models (uncomment to use)
# model_provider="custom_model"
# models=(
#   "qwen3_0_6b"
#   "qwen3_8b"
#   "qwen3_32b"
# )

# # OpenAI models (uncomment to use)
# model_provider="openai"
# models=(
#   "gpt-5-2025-08-07"
# )

# ========= Helper function to determine modes for each task =========
get_modes_for_task() {
  local task="$1"
  case "$task" in
    "relaxation_factor"|"residual_threshold")
      # These tasks use zero-shot mode only
      echo "-z"
      ;;
    *)
      # Other tasks use both modes
      echo "-z "
      ;;
  esac
}

# ========= Main loop =========
echo "🚀 Starting NS Transient 2D ICL variants inference and evaluation"
echo "📋 ICL datasets: ${icl_datasets[*]}"
echo "📋 Tasks: ${tasks[*]}"
echo "📋 Precision levels: ${precision_levels[*]}"
echo "📋 Models: ${models[*]}"
echo "📋 Provider: $model_provider"
echo ""

for dataset in "${icl_datasets[@]}"; do
  echo "💫 Processing dataset: $dataset"

  for task in "${tasks[@]}"; do
    echo "  🎯 Task: $task"

    # Get task-specific modes
    task_modes=$(get_modes_for_task "$task")

    for mode in $task_modes; do
      mode_name=$([ "$mode" = "-z" ] && echo "zero-shot" || echo "iterative")
      echo "    📊 Mode: $mode_name"

      for precision_level in "${precision_levels[@]}"; do
        echo "      🎛️  Precision: $precision_level"

        for model in "${models[@]}"; do
          echo "        🤖 Model: $model"

          # Run inference
          inference_cmd="python inference/langchain_LLM.py -p $model_provider -m $model -d $dataset -t $task -l $precision_level $mode --resume"
          run_cmd "$inference_cmd"

          # Run evaluation
          eval_cmd="python evaluation/ns_transient_2d/eval.py -m $model -d $dataset -t $task -l $precision_level $mode"
          run_cmd "$eval_cmd"

          echo "        ✅ Completed: $dataset $task $precision_level $mode_name $model"
        done
      done
    done
  done

  echo "✅ Completed all tasks for dataset: $dataset"
  echo ""
done

echo "🎉 All NS Transient 2D ICL variants inference and evaluation completed successfully!"
echo "📁 Results saved in: results_model_attempt/ns_transient_2d_icl*/precision_level/task/"
echo "📁 Logs saved in: log_model_tool_call/ns_transient_2d_icl*/precision_level/task/"
echo "📁 Evaluation logs saved in: eval_results/ns_transient_2d_icl*/task/precision_level/"