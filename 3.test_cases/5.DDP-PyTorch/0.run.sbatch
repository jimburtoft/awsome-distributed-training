#!/bin/bash

#SBATCH --exclusive  # job has exclusive use of the resource, no sharing
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --nodes 2 # number of nodes we want
set -ex;


###########################
###### User Variables #####
###########################

: "${IMAGE:=/apps/ddp-pytorch.sqsh}"
: "${TEST_CASE_PATH:=/home/ec2-user/2.PyTorch-DDP-validation}"

declare -a ARGS=(
    --container-image $IMAGE # Enroot squash file to use
    --container-mount-home # we mount the home directory where run.py is located
)


declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname):0 \
)

srun -l "${ARGS[@]}" torchrun \
   "${TORCHRUN_ARGS[@]}" \
    ${TEST_CASE_PATH}/1.run.py
