#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=2 # number of nodes to use
#SBATCH --job-name=llama3-fsdp-neuron
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive

set -ex;

###########################
###### User Variables #####
###########################

# Model recipe from ../../models (llama3_2_1b, llama3_1_8b, ...).
MODEL=${MODEL:-llama3_2_1b}
# One process per logical NeuronCore. trn2.48xlarge: 64, trn2.3xlarge: 4 (LNC=2, the default).
NPROC_PER_NODE=${NPROC_PER_NODE:-64}

export CONTAINER_IMAGE=${CONTAINER_IMAGE:-$(pwd)/pytorch-fsdp-neuron.sqsh}
export DATA_PATH=/fsx
export FSX_MOUNT=$(pwd):$DATA_PATH

###########################
## Environment Variables ##
###########################

# torch-neuronx's "neuron" process-group backend derives the Neuron runtime root address
# (NEURON_RT_ROOT_COMM_ID) from MASTER_ADDR, which torchrun sets from the rendezvous endpoint.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
## Set HuggingFace metadata timeout (in seconds) for large clusters
export HF_HUB_ETAG_TIMEOUT=60
## Streaming allenai/c4 anonymously is rate limited (HTTP 429); export HF_TOKEN before sbatch.
export HF_TOKEN=${HF_TOKEN:-}

declare -a ARGS=(
    --container-image $CONTAINER_IMAGE
    --container-mounts $FSX_MOUNT
    --container-env FI_PROVIDER,FI_EFA_USE_DEVICE_RDMA,FI_EFA_FORK_SAFE,HF_HUB_ETAG_TIMEOUT,HF_TOKEN
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$NPROC_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$MASTER_ADDR:29500
)

# Training arguments: the shared model recipe, with checkpoints written to FSx.
mapfile -t TRAINING_ARGS < <(sed -e 's/ *#.*//' -e '/^$/d' \
    -e "s|^--checkpoint_dir=.*|--checkpoint_dir=$DATA_PATH/checkpoints|" \
    -e "s|^--resume_from_checkpoint=.*|--resume_from_checkpoint=$DATA_PATH/checkpoints|" \
    "$(dirname "$(pwd)")/../models/${MODEL}.txt")

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi
srun ${AUTO_RESUME} -l "${ARGS[@]}" torchrun "${TORCHRUN_ARGS[@]}" /opt/fsdp/train.py "${TRAINING_ARGS[@]}"
