#!/bin/bash
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --wait-all-nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -ex;

RDV_ADDR=$(hostname)
WORLD_SIZE=$SLURM_JOB_NUM_NODES

export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa
export FI_EFA_ENABLE_SHM_TRANSFER=1

export NCCL_PROTO=simple
export NCCL_DEBUG=INFO

srun --container-image=/apps/param.sqsh -l torchrun \
   --nproc_per_node $SLURM_GPUS_PER_NODE \
   --nnodes $WORLD_SIZE \
   --rdzv_id $SLURM_JOB_ID \
   --rdzv_backend c10d \
   --rdzv_endpoint $RDV_ADDR \
   /param/train/comms/pt/comms.py --b=8 --e=2GB --f=2 --collective=all_reduce --num_iters=100