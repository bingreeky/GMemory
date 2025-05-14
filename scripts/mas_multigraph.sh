#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

# 加载 .env 文件（如果存在）
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT"

# 进入项目根目录
cd "$PROJECT_ROOT"

# 运行 Python 脚本
python3 tasks/run.py \
    --task alfworld \
    --reasoning io \
    --mas_memory multigraph \
    --max_trials 30 \
    --mas_type autogen \
    --model Qwen/Qwen2.5-14B-Instruct \
    # --model Qwen/Qwen2.5-14B-Instruct \
    # --model gpt-4o-mini \