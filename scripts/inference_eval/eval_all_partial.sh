#!/bin/bash
# eval_all.sh
# Function: Execute evaluation commands for specified dataset(s)
# Usage: ./eval_all.sh -d <dataset1> [-d <dataset2>] ...

# Removed 'set -e' to allow script to continue on errors
set -E -o pipefail

# Default values
DATASETS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model list (modify this list as needed)
models=(
#   "anthropic.claude-3-5-haiku-20241022-v1:0"
#   "anthropic.claude-3-5-sonnet-20240620-v1:0"
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
    echo "                   Available: burgers_1d, euler_1d, epoch_1d, heat_1d, heat_2d, euler_2d, mpm_2d, ns_2d, ns_transient_2d, diff_react_1d, hasegawa_mima_nonlinear, hasegawa_mima_linear, fem_2d"
    echo ""
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Available datasets and their tasks:"
    echo "  burgers_1d: cfl, beta, k, n_space (with precision levels: low, medium, high)"
    echo "  euler_1d: cfl, beta, k, n_space (with precision levels: low, medium, high)"
    echo "  epoch_1d: dt_multiplier, nx, npart, field_order, particle_order (with precision levels: low, medium, high)"
    echo "  heat_1d: cfl, n_space (with precision levels: low, medium, high)"
    echo "  heat_2d: dx, error_threshold (both modes), relax, t_init (zero-shot only) (with precision levels: low, medium, high)"
    echo "  euler_2d: n_grid_x, cfl, cg_tolerance (with precision levels: low, medium, high)"
    echo "  mpm_2d: nx, npart, cfl (with precision levels: low, medium, high)"
    echo "  ns_2d: mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold (with precision levels: low, medium, high)"
    echo "  ns_transient_2d: resolution, cfl, relaxation_factor, residual_threshold (with precision levels: low, medium, high)"
    echo "  diff_react_1d: cfl, n_space, tol (with precision levels: low, medium, high)"
    echo "  hasegawa_mima_nonlinear: N, dt (with precision levels: low, medium, high)"
    echo "  hasegawa_mima_linear: N, dt, cg_atol (with precision levels: low, medium, high)"
    echo "  fem_2d: dx, cfl (with precision levels: low, medium, high)"
    echo ""
    echo "Models to be evaluated:"
    for model in "${models[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Examples:"
    echo "  $0 -d burgers_1d"
    echo "  $0 -d heat_1d -d heat_2d"
    echo "  $0 -d burgers_1d -d euler_1d -d epoch_1d -d heat_1d -d heat_2d -d mpm_2d -d ns_2d -d ns_transient_2d"
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
            tasks=("cfl" "beta" "k" "n_space")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative
            
            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/burgers_1d/eval.py -m $MODEL -d burgers_1d -t $task -l $precision $mode"
                        if ! python evaluation/burgers_1d/eval.py -m $MODEL -d burgers_1d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
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
                        if ! python evaluation/euler_1d/eval.py -m $MODEL -d euler_1d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;

        "epoch_1d")
            echo "📋 Running Epoch 1D evaluation..."
            tasks=("dt_multiplier" "nx" "npart" "field_order" "particle_order")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/epoch_1d/eval.py -m $MODEL -d epoch_1d -t $task -l $precision $mode"
                        if ! python evaluation/epoch_1d/eval.py -m $MODEL -d epoch_1d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
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
                        if ! python evaluation/heat_1d/eval.py -m $MODEL -d heat_1d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
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
                        if ! python evaluation/heat_2d/eval.py -m $MODEL -d heat_2d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;

        "euler_2d")
            echo "📋 Running Euler 2D evaluation..."
            tasks=("n_grid_x" "cfl" "cg_tolerance")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/euler_2d/eval.py -m $MODEL -d euler_2d -t $task -l $precision $mode"
                        if ! python evaluation/euler_2d/eval.py -m $MODEL -d euler_2d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;

        "ns_2d")
            echo "📋 Running NS 2D evaluation..."
            tasks=("mesh_x" "mesh_y" "omega_u" "omega_v" "omega_p" "diff_u_threshold" "diff_v_threshold" "res_iter_v_threshold")
            precision_levels=("low" "medium" "high")
            
            for task in "${tasks[@]}"; do
                # Determine which modes to use for each task
                if [[ "$task" == "mesh_x" || "$task" == "mesh_y" ]]; then
                    # These tasks support both modes
                    task_modes=("-z" "")
                else
                    # Other tasks only support zero-shot mode
                    task_modes=("-z")
                fi
                
                for mode in "${task_modes[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/ns_2d/eval.py -m $MODEL -d ns_2d -t $task -l $precision $mode"
                        if ! python evaluation/ns_2d/eval.py -m $MODEL -d ns_2d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;
            
        "ns_transient_2d")
            echo "📋 Running NS Transient 2D evaluation..."
            tasks=("resolution" "cfl" "relaxation_factor" "residual_threshold")
            precision_levels=("low" "medium" "high")
            
            for task in "${tasks[@]}"; do
                # Determine which modes to use for each task
                if [[ "$task" == "relaxation_factor" || "$task" == "residual_threshold" ]]; then
                    # These tasks only support zero-shot mode
                    task_modes=("-z")
                else
                    # Other tasks support both modes
                    task_modes=("-z" "")
                fi
                
                for mode in "${task_modes[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/ns_transient_2d/eval.py -m $MODEL -d ns_transient_2d -t $task -l $precision $mode"
                        if ! python evaluation/ns_transient_2d/eval.py -m $MODEL -d ns_transient_2d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;

        "mpm_2d")
            echo "📋 Running MPM 2D evaluation..."
            tasks=("nx" "npart" "cfl")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/mpm_2d/eval.py -m $MODEL -d mpm_2d -t $task -l $precision $mode"
                        if ! python evaluation/mpm_2d/eval.py -m $MODEL -d mpm_2d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;

        "diff_react_1d")
            echo "📋 Running Diffusion-Reaction 1D evaluation..."
            tasks=("cfl" "n_space" "tol")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/diff_react_1d/eval.py -m $MODEL -d diff_react_1d -t $task -l $precision $mode"
                        if ! python evaluation/diff_react_1d/eval.py -m $MODEL -d diff_react_1d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;

        "hasegawa_mima_nonlinear")
            echo "📋 Running Hasegawa-Mima Nonlinear evaluation..."
            tasks=("N" "dt")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/hasegawa_mima_nonlinear/eval.py -m $MODEL -d hasegawa_mima_nonlinear -t $task -l $precision $mode"
                        if ! python evaluation/hasegawa_mima_nonlinear/eval.py -m $MODEL -d hasegawa_mima_nonlinear -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;

        "hasegawa_mima_linear")
            echo "📋 Running Hasegawa-Mima Linear evaluation..."
            tasks=("N" "dt" "cg_atol")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/hasegawa_mima_linear/eval.py -m $MODEL -d hasegawa_mima_linear -t $task -l $precision $mode"
                        if ! python evaluation/hasegawa_mima_linear/eval.py -m $MODEL -d hasegawa_mima_linear -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;

        "fem_2d")
            echo "📋 Running FEM 2D evaluation..."
            tasks=("dx" "cfl")
            precision_levels=("low" "medium" "high")
            modes=("-z" "")   # "-z" for zero-shot, empty string for iterative

            for mode in "${modes[@]}"; do
                for task in "${tasks[@]}"; do
                    for precision in "${precision_levels[@]}"; do
                        echo "▶ Executing: python evaluation/fem_2d/eval.py -m $MODEL -d fem_2d -t $task -l $precision $mode"
                        if ! python evaluation/fem_2d/eval.py -m $MODEL -d fem_2d -t $task -l $precision $mode; then
                            echo "⚠️  Evaluation failed for model: $MODEL, task: $task, precision: $precision, mode: ${mode:-iterative}. Continuing with next evaluation..."
                        fi
                    done
                done
            done
            ;;

        *)
            echo "❌ Unsupported dataset: $DATASET"
            echo "Supported datasets: burgers_1d, euler_1d, epoch_1d, heat_1d, heat_2d, euler_2d, mpm_2d, ns_2d, ns_transient_2d, diff_react_1d, hasegawa_mima_nonlinear, hasegawa_mima_linear, fem_2d"
            exit 1
            ;;
    esac
    
        echo "✅ Completed evaluation for model: $MODEL on dataset: $DATASET"
    done
    
    echo "✅ Completed all models for dataset: $DATASET"
done

echo ""
echo "🎉 All evaluation tasks completed successfully for all models on all specified datasets!"