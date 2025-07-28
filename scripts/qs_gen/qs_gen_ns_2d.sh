#!/bin/bash

# Define log file to track executed commands
LOG_FILE="scripts/qs_gen/ns_2d_execution_log.txt"

# Define array of commands to execute
commands=(
    "python qs_gen/2D_ns.py -n 25 -t mesh_x -z"
    "python qs_gen/2D_ns.py -n 25 -t mesh_x"
    "python qs_gen/2D_ns.py -n 25 -t mesh_y -z"
    "python qs_gen/2D_ns.py -n 25 -t mesh_y"
    "python qs_gen/2D_ns.py -n 25 -t omega_u -z"
    "python qs_gen/2D_ns.py -n 25 -t omega_u"
    "python qs_gen/2D_ns.py -n 25 -t omega_v -z"
    "python qs_gen/2D_ns.py -n 25 -t omega_v"
    "python qs_gen/2D_ns.py -n 25 -t omega_p -z"
    "python qs_gen/2D_ns.py -n 25 -t omega_p"
    "python qs_gen/2D_ns.py -n 25 -t diff_u_threshold -z"
    "python qs_gen/2D_ns.py -n 25 -t diff_u_threshold"
    "python qs_gen/2D_ns.py -n 25 -t diff_v_threshold -z"
    "python qs_gen/2D_ns.py -n 25 -t diff_v_threshold"
    "python qs_gen/2D_ns.py -n 25 -t res_iter_v_threshold -z"
    "python qs_gen/2D_ns.py -n 25 -t res_iter_v_threshold"
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

echo "Running 2D Navier-Stokes Channel Flow experiments..."
echo "Note: Each -n 25 generates 25 profiles per boundary condition (total: 1 + 4×25 = 101 profiles)"

# Iterate through commands and execute
for ((i = start_index; i < ${#commands[@]}; i++)); do
    echo "Executing: ${commands[$i]}"
    
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
done

echo "Done."