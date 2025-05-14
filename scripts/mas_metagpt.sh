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
    --mas_memory metagpt \
    --max_trials 30 \
    --mas_type dylan \
    --model Pro/Qwen/Qwen2.5-7B-Instruct \
    # --model Qwen/Qwen2.5-14B-Instruct \
    # --model gpt-4o-mini \
    