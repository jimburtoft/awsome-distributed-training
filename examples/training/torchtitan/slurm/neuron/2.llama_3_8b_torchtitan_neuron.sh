#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=llama3_8b_torchtitan_neuron
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out

set -ex;

# One process per logical NeuronCore. trn2.48xlarge: 64 (LNC=2, the default).
NPROC_PER_NODE=${NPROC_PER_NODE:-64}
# torchtitan-neuron recipe (see torchtitan_neuron/docs/dense-tutorials.md in the image):
#   llama3_8b_sac_eager, llama3_8b_sac_compile, llama3_8b_fullac_compile, ...
CONFIG=${CONFIG:-llama3_8b_sac_compile}
MODULE=${MODULE:-torchtitan_neuron.models.llama3}
TP=${TP:-4}
# Data-parallel replicas across nodes (HSDP). 1 = FSDP across all nodes.
DP_REPLICATE=${DP_REPLICATE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$(( 16 * SLURM_JOB_NUM_NODES ))}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export HF_HUB_ETAG_TIMEOUT=60

declare -a ARGS=(
    --container-image $(pwd)/torchtitan-neuron.sqsh
    --container-mounts $(pwd)/assets:/workspace/torchtitan_neuron/assets/hf,$(pwd)/outputs:/workspace/torchtitan_neuron/outputs
    --container-env FI_PROVIDER,FI_EFA_USE_DEVICE_RDMA,FI_EFA_FORK_SAFE,HF_HUB_ETAG_TIMEOUT
)

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$NPROC_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$MASTER_ADDR:29500
    --local-ranks-filter 0
)

declare -a TRAIN_ARGS=(
    --module ${MODULE}
    --config ${CONFIG}
    --parallelism.tensor_parallel_degree=${TP}
    --parallelism.data_parallel_replicate_degree=${DP_REPLICATE}
    --parallelism.data_parallel_shard_degree=-1
    --parallelism.context_parallel_degree=1
    --training.global_batch_size=${GLOBAL_BATCH_SIZE}
)

mkdir -p outputs
AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi
srun ${AUTO_RESUME} -l "${ARGS[@]}" torchrun "${TORCHRUN_ARGS[@]}" -m torchtitan.train "${TRAIN_ARGS[@]}" "$@"
