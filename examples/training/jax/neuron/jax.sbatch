#!/bin/bash
# SPMD LM training with JAX NeuronX on Trainium (trn2) nodes, one task per node.
#SBATCH --job-name=jax-neuron
#SBATCH -o jax_neuron_%j.out
#SBATCH -e jax_neuron_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

set -euo pipefail

IMAGE=${IMAGE:-/fsx/jax-neuron.sqsh}  # enroot import of the image built from neuron/Dockerfile
CODE_DIR=${CODE_DIR:-$(pwd)}
CONFIG=${CONFIG:-1b}
MODEL_PARALLEL=${MODEL_PARALLEL:-4}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-4}
SEQ=${SEQ:-1024}
STEPS=${STEPS:-300}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NNODES=$SLURM_JOB_NUM_NODES

srun --container-image "$IMAGE" --container-mounts "$CODE_DIR":/opt/jaxlm --no-container-mount-home \
  --no-container-entrypoint --export=ALL bash -c 'NODE_RANK=$SLURM_NODEID /opt/jaxlm/run_neuron.sh \
    --config '"$CONFIG"' --model-parallel '"$MODEL_PARALLEL"' --per-device-batch '"$PER_DEVICE_BATCH"' \
    --seq '"$SEQ"' --steps '"$STEPS"' --log-every 25'
