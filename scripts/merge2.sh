#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"

python3 full_tasks/run.py \
    --task fever \
    --reasoning io \
    --mas_memory empty \
    --max_trials 15 \
    --mas_type dylan \
    --model Qwen/Qwen2.5-72B-Instruct \

python3 full_tasks/run.py \
    --task pddl \
    --reasoning io \
    --mas_memory empty \
    --max_trials 30 \
    --mas_type dylan \
    --model Qwen/Qwen2.5-72B-Instruct \