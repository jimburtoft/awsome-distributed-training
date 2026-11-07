#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=picotron-neuron
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out

set -ex;

# One process per logical NeuronCore. trn2.48xlarge: 64, trn2.3xlarge: 4 (LNC=2, the default).
NPROC_PER_NODE=${NPROC_PER_NODE:-64}
# Config created with: python create_config.py --use_neuron --out_dir conf --exp_name <name> ...
CONFIG=${CONFIG:-$PWD/conf/smollm-neuron/config.json}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1

declare -a SRUN_ARGS=(
    --container-image $PWD/picotron-neuron.sqsh
    --container-mounts "$PWD"
    --container-workdir ${PWD}
    --container-env FI_PROVIDER,FI_EFA_USE_DEVICE_RDMA,FI_EFA_FORK_SAFE,HF_TOKEN
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$NPROC_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$MASTER_ADDR:29500
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi
srun ${AUTO_RESUME} -l "${SRUN_ARGS[@]}" torchrun "${TORCHRUN_ARGS[@]}" /opt/picotron/train.py --config ${CONFIG}
