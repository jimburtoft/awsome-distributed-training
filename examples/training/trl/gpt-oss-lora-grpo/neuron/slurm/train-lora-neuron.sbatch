#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# GPT-OSS LoRA SFT on AWS Trainium (trn2) with Slurm (SageMaker HyperPod or ParallelCluster) and Pyxis.
# Same arguments as neuron/train-lora-neuron.yaml-template. Run from this example's directory:
#   docker build -f neuron/Dockerfile -t gpt-oss-neuron . && enroot import -o gpt-oss-neuron.sqsh dockerd://gpt-oss-neuron:latest
#   sbatch --nodes 1 neuron/slurm/train-lora-neuron.sbatch

#SBATCH --job-name=gpt-oss-lora-neuron
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --output=logs/%x_%j.out

set -ex

# gpt-oss-20b is dequantized from MXFP4 on the host by every rank (about 35 GB of host RAM each), so the
# default is 16 ranks per node (one per Neuron device), which fits in 2 TiB.
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
MODEL=${MODEL:-openai/gpt-oss-20b}
IMAGE=${IMAGE:-$(pwd)/gpt-oss-neuron.sqsh}
OUTPUT_DIR=${OUTPUT_DIR:-/fsx/${USER}/gpt-oss-lora-neuron}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export HF_TOKEN=${HF_TOKEN:-}
# Shared Hugging Face cache on FSx, so ranks do not each download from the Hub.
export HF_HOME=${HF_HOME:-/fsx/${USER}/.cache/huggingface}
mkdir -p "$HF_HOME" "$OUTPUT_DIR" logs

declare -a ARGS=(
    --container-image ${IMAGE}
    --container-mounts ${OUTPUT_DIR}:${OUTPUT_DIR},${HF_HOME}:${HF_HOME}
    --container-env FI_PROVIDER,FI_EFA_USE_DEVICE_RDMA,FI_EFA_FORK_SAFE,HF_TOKEN,HF_HOME
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=${NPROC_PER_NODE}
    --nnodes=${SLURM_JOB_NUM_NODES}
    --rdzv_id=${SLURM_JOB_ID}
    --rdzv_backend=c10d
    --rdzv_endpoint=${MASTER_ADDR}:29500
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun ${AUTO_RESUME} -l "${ARGS[@]}" torchrun "${TORCHRUN_ARGS[@]}" /opt/gpt-oss/sft.py \
    --config /opt/gpt-oss/configs/sft_lora.yaml \
    --model_name_or_path ${MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --bf16=True \
    --per_device_train_batch_size=1 \
    --gradient_checkpointing=true \
    --packing=true \
    --packing_strategy=wrapped \
    --max_length=1024 \
    --save_steps=25 \
    --save_total_limit=5 \
    --logging_steps=1 \
    --report_to=none \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config '{"version": 2, "transformer_layer_cls_to_wrap": ["GptOssDecoderLayer"], "cpu_ram_efficient_loading": true}' \
    "$@"
