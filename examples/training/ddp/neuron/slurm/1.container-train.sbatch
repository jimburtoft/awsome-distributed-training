#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=ddp-neuron
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --nodes 2
#SBATCH --output=logs/%x_%j.out # logfile for stdout/stderr

export LOGLEVEL=INFO

# torch-neuronx's "neuron" process-group backend derives the Neuron runtime root address
# (NEURON_RT_ROOT_COMM_ID) from MASTER_ADDR, which torchrun sets from the rendezvous endpoint.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# EFA for cross-node collectives.
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1

# One process per logical NeuronCore. trn2.3xlarge: 4, trn2.48xlarge: 64 (LNC=2, the default).
NPROC_PER_NODE=${NPROC_PER_NODE:-64}

declare -a ARGS=(
    --container-image ${PWD}/pytorch-neuron.sqsh
    --container-mounts ${PWD}:/output
    --container-workdir /output
    --container-env FI_PROVIDER,FI_EFA_USE_DEVICE_RDMA,FI_EFA_FORK_SAFE
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node="$NPROC_PER_NODE"
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$MASTER_ADDR:29500
)

declare -a TRAIN_ARGS=(
    --total_epochs 500
    --save_every 1
    --batch_size 32
    --checkpoint_path /output/snapshot.pt
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi

srun ${AUTO_RESUME} -l "${ARGS[@]}" torchrun \
    "${TORCHRUN_ARGS[@]}" \
    /opt/ddp/ddp.py ${TRAIN_ARGS[@]}
