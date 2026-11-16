#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=qwen3-8b-lora-neuron
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out

set -ex;

# One process per logical NeuronCore. trn2.48xlarge: 64, trn2.3xlarge: 4 (LNC=2, the default).
NPROC_PER_NODE=${NPROC_PER_NODE:-64}
IMAGE=${IMAGE:-$(pwd)/qlora-neuron.sqsh}
OUTPUT_DIR=${OUTPUT_DIR:-/fsx/${USER}/qwen3-8b-lora-neuron}
CONFIG=${CONFIG:-configs/training_config_zero2.yaml}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export HF_TOKEN=${HF_TOKEN:-}
# Shared Hugging Face cache on FSx, so ranks do not each download from the Hub.
export HF_HOME=${HF_HOME:-/fsx/${USER}/.cache/huggingface}
mkdir -p "$HF_HOME"

mkdir -p ${OUTPUT_DIR}
declare -a ARGS=(
    --container-image ${IMAGE}
    --container-mounts ${OUTPUT_DIR}:${OUTPUT_DIR},${HF_HOME}:${HF_HOME},$(pwd)/configs:/opt/qlora/configs
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
srun ${AUTO_RESUME} -l "${ARGS[@]}" torchrun "${TORCHRUN_ARGS[@]}" \
    src/train_lora_fsdp2.py --config_path ${CONFIG} --output_dir ${OUTPUT_DIR} "$@"
