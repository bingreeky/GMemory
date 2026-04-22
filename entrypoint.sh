#!/bin/bash
set -e

echo "Starting job..."

echo "TASK: $TASK"
echo "MEMORY: $MEMORY"
echo "MODEL: $MODEL_NAME"
echo "SEED: $SEED"

python tasks/run.py --task "$TASK" --reasoning io --mas_memory "$MEMORY" --mas_type autogen --model "$MODEL_NAME" --seed "$SEED"

echo "Ending job..."
