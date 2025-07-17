#!/bin/bash

# 定义一个日志文件，用来记录已执行的命令
LOG_FILE="2d_heat_transfer_execution_log.txt"

# 定义要执行的命令数组
commands=(
    "python qs_gen/2D_heat_transfer.py -n 10 -t dx -z"
    "python qs_gen/2D_heat_transfer.py -n 10 -t dx"
    "python qs_gen/2D_heat_transfer.py -n 10 -t error_threshold -z"
    "python qs_gen/2D_heat_transfer.py -n 10 -t error_threshold"
    "python qs_gen/2D_heat_transfer.py -n 10 -t relax"
    "python qs_gen/2D_heat_transfer.py -n 10 -t t_init"
)

# 检查是否有日志文件，并获取最后执行的命令索引
if [ -f "$LOG_FILE" ]; then
    last_command=$(tail -n 1 "$LOG_FILE")
    # 查找命令数组中最后执行的命令索引
    for i in "${!commands[@]}"; do
        if [ "${commands[$i]}" == "$last_command" ]; then
            last_command_index=$i
            break
        fi
    done
    # 从下一个命令开始执行
    start_index=$((last_command_index + 1))
else
    # 如果没有日志文件，则从第一个命令开始执行
    start_index=0
fi

echo "Running 2D_heat_transfer experiments..."

# 遍历命令并执行
for ((i = start_index; i < ${#commands[@]}; i++)); do
    echo "Executing: ${commands[$i]}"
    
    # 执行命令
    ${commands[$i]}
    
    # 检查命令是否成功执行，如果失败则退出脚本
    if [ $? -ne 0 ]; then
        echo "Error occurred during execution of: ${commands[$i]}"
        echo "Stopping the execution."
        exit 1
    fi
    
    # 记录已执行的命令到日志文件
    echo "${commands[$i]}" >> "$LOG_FILE"
done

echo "Done."
