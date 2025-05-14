#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"

python3 tasks/run.py \
    --task pddl \
    --reasoning io \
    --mas_memory chatdev \
    --max_trials 30 \
    --mas_type autogen \
    --model Qwen/Qwen2.5-72B-Instruct \
    # --model gpt-4o-mini \
    # --model Qwen/Qwen2.5-14B-Instruct \