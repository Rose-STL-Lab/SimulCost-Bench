#!/bin/bash

# qs_gen_all.sh - Execute multiple qs_gen scripts
# Usage: ./qs_gen_all.sh [script1] [script2] ... [scriptN]
# Example: ./qs_gen_all.sh burgers_1d euler_1d heat_1d heat_2d

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_SCRIPTS=("burgers_1d" "euler_1d" "heat_1d" "heat_2d")

# Function to display usage
usage() {
    echo "Usage: $0 [script1] [script2] ... [scriptN]"
    echo "Available scripts:"
    for script in "${ALL_SCRIPTS[@]}"; do
        echo "  - ${script}"
    done
    echo ""
    echo "Examples:"
    echo "  $0 burgers_1d euler_1d    # Execute only burgers_1d and euler_1d"
    echo "  $0                        # Execute all available scripts"
    exit 1
}

# Function to execute a single script
execute_script() {
    local script_name="$1"
    local script_path="${SCRIPT_DIR}/qs_gen_${script_name}.sh"
    
    if [ ! -f "$script_path" ]; then
        echo "Error: Script '${script_path}' not found!"
        return 1
    fi
    
    if [ ! -x "$script_path" ]; then
        echo "Making script executable: ${script_path}"
        chmod +x "$script_path"
    fi
    
    echo "========================================"
    echo "Starting execution of: ${script_name}"
    echo "========================================"
    
    # Execute the script
    if bash "$script_path"; then
        echo "✓ Successfully completed: ${script_name}"
        return 0
    else
        echo "✗ Failed to execute: ${script_name}"
        return 1
    fi
}

# Main execution logic
main() {
    local scripts_to_run=()
    
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        # No arguments provided, run all scripts
        scripts_to_run=("${ALL_SCRIPTS[@]}")
        echo "No scripts specified. Running all available scripts..."
    else
        # Handle help option
        if [[ "$1" == "-h" || "$1" == "--help" ]]; then
            usage
        fi
        
        # Validate provided script names
        for arg in "$@"; do
            if [[ " ${ALL_SCRIPTS[*]} " =~ " ${arg} " ]]; then
                scripts_to_run+=("$arg")
            else
                echo "Error: Unknown script '${arg}'"
                echo "Available scripts: ${ALL_SCRIPTS[*]}"
                exit 1
            fi
        done
    fi
    
    echo "Scripts to execute: ${scripts_to_run[*]}"
    echo ""
    
    # Execute scripts sequentially
    local total_scripts=${#scripts_to_run[@]}
    local completed_scripts=0
    local failed_scripts=()
    
    for script in "${scripts_to_run[@]}"; do
        if execute_script "$script"; then
            ((completed_scripts++))
        else
            failed_scripts+=("$script")
        fi
        echo ""
    done
    
    # Summary
    echo "========================================"
    echo "EXECUTION SUMMARY"
    echo "========================================"
    echo "Total scripts: ${total_scripts}"
    echo "Completed successfully: ${completed_scripts}"
    echo "Failed: ${#failed_scripts[@]}"
    
    if [ ${#failed_scripts[@]} -gt 0 ]; then
        echo "Failed scripts: ${failed_scripts[*]}"
        exit 1
    else
        echo "All scripts completed successfully!"
        exit 0
    fi
}

# Run main function with all arguments
main "$@"