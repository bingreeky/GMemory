#!/bin/bash

if [ -f "./.env" ]; then
    export $(grep -v '^#' "./.env" | xargs)
fi

# Options:
# --mas_memory:    empty, chatdev, metagpt, voyager, generative, memorybank, g-memory
# --mas_type:      autogen, dylan, graph
# --use_projector  store_true

python3 tasks/run.py \
    --task alfworld \
    --reasoning io \
    --mas_memory g-memory \
    --max_trials 30 \
    --mas_type dylan \
    --model Pro/Qwen/Qwen2.5-7B-Instruct \