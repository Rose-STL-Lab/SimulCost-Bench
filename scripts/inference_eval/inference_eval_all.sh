#!/bin/bash
# inference_eval_all.sh
# Function: Execute inference and evaluation scripts for all tasks sequentially

echo "🚀 Starting inference and evaluation for all tasks..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define all inference scripts
scripts=(
    "$SCRIPT_DIR/inference_eval_burgers_1d.sh"
    "$SCRIPT_DIR/inference_eval_euler_1d.sh"
    "$SCRIPT_DIR/inference_eval_heat_1d.sh"
    "$SCRIPT_DIR/inference_eval_heat_2d.sh"
)

# Execute each script sequentially
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "📋 Executing: $(basename "$script")"
        bash "$script"
        
        if [ $? -eq 0 ]; then
            echo "✅ $(basename "$script") executed successfully"
        else
            echo "❌ $(basename "$script") execution failed"
            exit 1
        fi
        echo ""
    else
        echo "⚠️ Script does not exist: $script"
        exit 1
    fi
done

echo "🎉 All inference and evaluation tasks completed!"