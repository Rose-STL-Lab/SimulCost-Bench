#!/bin/bash
# run_all_resume.sh
# 功能：按组合顺序执行全部命令，遇错终止；下次运行从出错命令继续
set -eE -o pipefail          # 任一命令或管道出错立即退出，保留ERR信息

RESUME_LOG=".resume_progress.log"   # 记录已成功命令
touch "$RESUME_LOG"

# ========= 通用执行函数 =========
run_cmd () {
  local cmd="$1"
  # 如果该命令完整行已记录，则跳过
  if grep -Fxq "$cmd" "$RESUME_LOG"; then
    echo "✔ 已完成，跳过: $cmd"
    return
  fi

  echo "▶ 执行: $cmd"
  eval "$cmd"               # 运行实际命令

  # 运行成功才追加到日志；若失败会触发 set -eE 直接退出，不写日志
  echo "$cmd" >> "$RESUME_LOG"
}

# ========= 参数列表 =========
tasks=("cfl" "k" "w")
models=(
  "anthropic.claude-3-5-haiku-20241022-v1:0"
  "anthropic.claude-3-5-sonnet-20240620-v1:0"
  "anthropic.claude-3-7-sonnet-20250219-v1:0"
)
cases=("blast" "double_shock" "rarefaction" "sin" "sod")
modes=("-z" "")   # "-z" 为 zero-shot，空串为 iterative

# ========= 主循环 =========
for mode in "${modes[@]}"; do
  for task in "${tasks[@]}"; do
    run_cmd "python dataset_gen/oneD_burgers.py -t $task $mode"

    for model in "${models[@]}"; do
      for case in "${cases[@]}"; do
        run_cmd "python inference/langchain_LLM.py -p bedrock -m $model -d burgers_1d -t $task -c $case $mode"
        run_cmd "PYTHONPATH=$(pwd) python evaluation/burgers/eval.py -m $model -d burgers_1d -t $task -c $case $mode"
      done
    done
  done
done

echo "✅ 所有任务顺利完成！"
