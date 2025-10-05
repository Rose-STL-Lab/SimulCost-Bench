#!/bin/bash
# inference_eval_euler_1d_icl.sh
# Function: Execute 1D Euler ICL dataset model inference and evaluation in sequential order, stop on error; resume from failed command on next run
set -eE -o pipefail          # Exit immediately on any command or pipeline error, preserve ERR information

RESUME_LOG="scripts/inference_eval/additional_exp/euler_1d_icl_resume_progress.log"   # Log file for successful commands
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
icl_datasets=("euler_1d_icl_accuracy_focused" "euler_1d_icl_cost_excluded" "euler_1d_icl_full")
tasks=("cfl" "beta" "k" "n_space")
precision_levels=("low" "medium" "high")
modes=("-z" "")   # "-z" for zero-shot, empty string for iterative
# modes=("-z")

# ========= Model provider configurations =========
declare -A config_bedrock_provider=(["provider"]="bedrock")
declare -a config_bedrock_models=(
  "anthropic.claude-3-7-sonnet-20250219-v1:0"
  "mistral.mistral-large-2402-v1:0"
  "meta.llama3-70b-instruct-v1:0"
  "amazon.nova-premier-v1:0"
)

declare -A config_openai_provider=(["provider"]="openai")
declare -a config_openai_models=(
  "gpt-5-2025-08-07"
)

declare -A config_custom_provider=(["provider"]="custom_model")
declare -a config_custom_models=(
  "qwen3_32b"
)

# List of all configurations
config_names=("bedrock" "openai" "custom")

# ========= Main loop =========
for config_name in "${config_names[@]}"; do
  # Set current provider and models based on config
  case $config_name in
    "bedrock")
      model_provider="${config_bedrock_provider[provider]}"
      models=("${config_bedrock_models[@]}")
      ;;
    "openai")
      model_provider="${config_openai_provider[provider]}"
      models=("${config_openai_models[@]}")
      ;;
    "custom")
      model_provider="${config_custom_provider[provider]}"
      models=("${config_custom_models[@]}")
      ;;
  esac

  echo "🔄 Starting evaluation for provider: $model_provider"

  for dataset in "${icl_datasets[@]}"; do
    echo "  📊 Processing dataset: $dataset"

    for mode in "${modes[@]}"; do
      for task in "${tasks[@]}"; do
        for model in "${models[@]}"; do
          for precision_level in "${precision_levels[@]}"; do
            run_cmd "python inference/langchain_LLM.py -p $model_provider -m $model -d $dataset -t $task -l $precision_level $mode --resume"
            run_cmd "python evaluation/euler_1d/eval.py -m $model -d $dataset -t $task -l $precision_level $mode"
          done
        done
      done
    done

    echo "  ✅ Completed dataset: $dataset"
  done

  echo "✅ Completed evaluation for provider: $model_provider"
done

echo "✅ All 1D Euler ICL equation inference tasks completed successfully!"