#!/bin/bash

# Dataset Generation Script for All Tasks
# This script generates datasets for all available tasks in the SimulCost-Bench project

# Define log file to track executed commands
LOG_FILE="scripts/ds_gen/ds_gen_all_execution_log.txt"

# Define array of commands to execute
commands=(
    # 1D Burgers equation tasks
    "python dataset_gen/oneD_burgers.py -t cfl -z"
    "python dataset_gen/oneD_burgers.py -t cfl"
    "python dataset_gen/oneD_burgers.py -t k -z"
    "python dataset_gen/oneD_burgers.py -t k"
    "python dataset_gen/oneD_burgers.py -t w -z"
    "python dataset_gen/oneD_burgers.py -t w"
    
    # 1D Euler equation tasks
    "python dataset_gen/oneD_euler.py -t cfl -z"
    "python dataset_gen/oneD_euler.py -t cfl"
    "python dataset_gen/oneD_euler.py -t beta -z"
    "python dataset_gen/oneD_euler.py -t beta"
    "python dataset_gen/oneD_euler.py -t k -z"
    "python dataset_gen/oneD_euler.py -t k"
    
    # 1D Heat Transfer tasks
    "python dataset_gen/oneD_heat_transfer.py -n 100 -t cfl -z"
    "python dataset_gen/oneD_heat_transfer.py -n 100 -t cfl"
    "python dataset_gen/oneD_heat_transfer.py -n 100 -t n_space -z"
    "python dataset_gen/oneD_heat_transfer.py -n 100 -t n_space"
    
    # 2D Heat Transfer tasks
    "python dataset_gen/twoD_heat_transfer.py -n 100 -t dx -z"
    "python dataset_gen/twoD_heat_transfer.py -n 100 -t dx"
    "python dataset_gen/twoD_heat_transfer.py -n 100 -t error_threshold -z"
    "python dataset_gen/twoD_heat_transfer.py -n 100 -t error_threshold"
    "python dataset_gen/twoD_heat_transfer.py -n 100 -t relax"
    "python dataset_gen/twoD_heat_transfer.py -n 100 -t t_init"
)

# Check for existing log file and get index of last executed command
if [ -f "$LOG_FILE" ]; then
    last_command=$(tail -n 1 "$LOG_FILE")
    # Find index of last executed command in commands array
    for i in "${!commands[@]}"; do
        if [ "${commands[$i]}" == "$last_command" ]; then
            last_command_index=$i
            break
        fi
    done
    # Start execution from next command
    start_index=$((last_command_index + 1))
else
    # If no log file exists, start from first command
    start_index=0
fi

echo "Starting dataset generation for all tasks..."
echo "Total commands to execute: ${#commands[@]}"

# Iterate through commands and execute
for ((i = start_index; i < ${#commands[@]}; i++)); do
    echo "===========================================" 
    echo "Executing command $((i + 1))/${#commands[@]}: ${commands[$i]}"
    echo "==========================================="
    
    # Execute command
    ${commands[$i]}
    
    # Check if command executed successfully, exit script if failed
    if [ $? -ne 0 ]; then
        echo "Error occurred during execution of: ${commands[$i]}"
        echo "Stopping the execution."
        exit 1
    fi
    
    # Log executed command to file
    echo "${commands[$i]}" >> "$LOG_FILE"
    
    echo "Command completed successfully."
    echo ""
done

echo "All dataset generation tasks completed successfully!"