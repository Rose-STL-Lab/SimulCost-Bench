#!/bin/bash

# 设置日志文件路径
LOG_FILE="utils/rsync_backup.log"

# 写入日志函数
log_message() {
    local DATE=$(date '+%Y-%m-%d %H:%M:%S')
    local MESSAGE="[$DATE] $1"
    echo "$MESSAGE" | tee -a "$LOG_FILE"
}

# 错误处理函数
error_exit() {
    log_message "ERROR: $1"
    echo "ERROR: $1"
    exit 1
}

# 目录同步函数
sync_directory() {
    local source_dir="$1"
    local target_dir="$2"
    
    log_message "Starting rsync from $source_dir to $target_dir"
    
    if [ ! -d "$source_dir" ]; then
        error_exit "Source directory $source_dir does not exist."
    fi
    
    rsync -av --progress "$source_dir" "$target_dir" 2>&1 | tee -a "$LOG_FILE"
    if [ $? -ne 0 ]; then
        error_exit "rsync failed for $source_dir to $target_dir."
    fi
    
    log_message "Rsync completed successfully from $source_dir to $target_dir"
}

# 执行同步操作
log_message "Starting rsync operations."

sync_directory "log_model_tool_call/" "/data/leo_work_new/log_model_tool_call/"
sync_directory "results_model_attempt/" "/data/leo_work_new/results_model_attempt/"
# sync_directory "eval_results/" "/data/leo_work_new/eval_results/"
# sync_directory "sim_res/" "/data/leo_work_new/sim_res/"

log_message "Rsync operations completed successfully."

# 结束脚本
exit 0
