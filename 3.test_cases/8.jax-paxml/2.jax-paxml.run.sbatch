#!/bin/bash
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=8
set -x;

EXP=${1:-LmCloudSpmd2B}
CONTAINER=/apps/jax-paxml.sqsh

SHARED_SOURCE=$(pwd)/shared/${SLURM_JOB_ID}
SHARED_DEST=/data
SCRIPTS_SOURCE=$(pwd)
SCRIPTS_DEST=/apps/jax-paxml

export NCCL_DEBUG=INFO
mkdir ${SHARED_SOURCE}

echo "start at: $(date)"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"

ENROOT_RESTRICT_DEV=n srun \
    --container-image="$CONTAINER" \
    --container-mounts=${SHARED_SOURCE}:${SHARED_DEST},${SCRIPTS_SOURCE}:${SCRIPTS_DEST} \
    --export=ALL,N_GPUS=$((${SLURM_NTASKS})),TFDS_DATA_DIR=${SHARED_DEST}/data,GPT_VOCAB_PATH=${SHARED_DEST}/vocab,LOG_DIR=${SHARED_DEST}/log \
    -l \
    ${SCRIPTS_DEST}/1.run-script.sh ${EXP}
echo "end at: $(date)"