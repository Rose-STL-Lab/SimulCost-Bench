#!/bin/bash
# eval_all.sh
# Function: Execute evaluation commands for specified dataset(s)
# Usage: ./eval_all.sh -d <dataset1> [-d <dataset2>] ...

set -eE -o pipefail

# Default values
DATASETS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model list (modify this list as needed)
models=(
  "anthropic.claude-3-5-haiku-20241022-v1:0"
  "anthropic.claude-3-5-sonnet-20240620-v1:0"
  "anthropic.claude-3-7-sonnet-20250219-v1:0"
  "mistral.mistral-large-2402-v1:0"
  "meta.llama3-70b-instruct-v1:0"
  "amazon.nova-premier-v1:0"
#   "qwen3_8b"
#   "qwen3_32b"
#   "qwen3_235b_a22b"
)

# Help function
show_help() {
    echo "Usage: $0 -d <dataset1> [-d <dataset2>] ..."
    echo ""
    echo "Required options:"
    echo "  -d, --dataset    Dataset name (can be specified multiple times)"
    echo "                   Available: burgers_1d, euler_1d, heat_1d, heat_2d"
    echo ""
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Available datasets and their tasks:"
    echo "  burgers_1d: cfl, k, w (with cases: blast, double_shock, rarefaction, sin, sod)"
    echo "  euler_1d: cfl, beta, k, n_space (with precision levels: low, medium, high)"
    echo "  heat_1d: cfl, n_space (with precision levels: low, medium, high)"
    echo "  heat_2d: dx, error_threshold (both modes), relax, t_init (zero-shot only) (with precision levels: low, medium, high)"
    echo ""
    echo "Models to be evaluated:"
    for model in "${models[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Examples:"
    echo "  $0 -d burgers_1d"
    echo "  $0 -d heat_1d -d heat_2d"
    echo "  $0 -d burgers_1d -d euler_1d -d heat_1d -d heat_2d"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASETS+=("$2")
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "Error: At least one dataset (-d) is required"
    show_help
    exit 1
fi

echo "🚀 Starting evaluation for ${#models[@]} model(s) on dataset(s): ${DATASETS[*]}"

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "🎯 Processing dataset: $DATASET"
    
    # Loop through each model
    for MODEL in "${models[@]}"; do
        echo ""
        echo "🔄 Processing model: $MODEL on dataset: $DATASET"
        
        # Execute based on dataset type
        case "$DATASET" in
        "burgers_1d")
            echo "📋 Running Burgers 1D evaluation..."
            tasks=("cfl" "k" "w")
            cases=("blast" "double_shock" "rarefaction" "sin" "sod")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative
            
            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for case in "${cases[@]}"; do
                        echo "▶ Executing: python evaluation/burgers/eval.py -m $MODEL -d burgers_1d -t $task -c $case $mode"
                        python evaluation/burgers/eval.py -m $MODEL -d burgers_1d -t $task -c $case $mode
                    done
                done
            done
            ;;
            
        "euler_1d")
            echo "📋 Running Euler 1D evaluation..."
            tasks=("cfl" "beta" "k" "n_space")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative
            
            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/euler_1d/eval.py -m $MODEL -d euler_1d -t $task -l $precision $mode"
                        python evaluation/euler_1d/eval.py -m $MODEL -d euler_1d -t $task -l $precision $mode
                    done
                done
            done
            ;;
            
        "heat_1d")
            echo "📋 Running Heat 1D evaluation..."
            tasks=("cfl" "n_space")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative
            
            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/heat_1d/eval.py -m $MODEL -d heat_1d -t $task -l $precision $mode"
                        python evaluation/heat_1d/eval.py -m $MODEL -d heat_1d -t $task -l $precision $mode
                    done
                done
            done
            ;;
            
        "heat_2d")
            echo "📋 Running Heat 2D evaluation..."
            # Define tasks and their corresponding modes
            declare -A task_modes
            task_modes["dx"]="-z "           # dx supports zero-shot and iterative
            task_modes["error_threshold"]="-z "  # error_threshold supports zero-shot and iterative  
            task_modes["relax"]="-z"         # relax only supports zero-shot
            task_modes["t_init"]="-z"        # t_init only supports zero-shot
            
            precision_levels=("low" "medium" "high")
            
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
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/heat_2d/eval.py -m $MODEL -d heat_2d -t $task -l $precision $mode"
                        python evaluation/heat_2d/eval.py -m $MODEL -d heat_2d -t $task -l $precision $mode
                    done
                done
            done
            ;;
            
        *)
            echo "❌ Unsupported dataset: $DATASET"
            echo "Supported datasets: burgers_1d, euler_1d, heat_1d, heat_2d"
            exit 1
            ;;
    esac
    
        echo "✅ Completed evaluation for model: $MODEL on dataset: $DATASET"
    done
    
    echo "✅ Completed all models for dataset: $DATASET"
done

echo ""
echo "🎉 All evaluation tasks completed successfully for all models on all specified datasets!"