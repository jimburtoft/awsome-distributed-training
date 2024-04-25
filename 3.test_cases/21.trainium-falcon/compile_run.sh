#!/usr/bin/env bash
export TASK_NAME=mrpc
export NEURON_CC_FLAGS="--model-type=transformer"
XLA_USE_BF16=1 python3 ./scripts/train.py \
        --gradient_checkpointing True \
        --bf16 True \
        --optimizer "adamw_torch" \
        --per_device_train_batch_size 1 \
        --epochs 1 \
        --max_steps 1 \
        --dataset_path "processed/data" \
        --fsdp "full_shard auto_wrap"|& tee log_run

