#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# GRPO on AWS Trainium (trn2.48xlarge): the last node serves rollouts with vLLM on AWS Neuron, the others
# train with PyTorch Native + FSDP2. Policy weights move through a shared FSx directory
# (neuron/trl_neuron_weight_sync.py) instead of NCCL.
#SBATCH --job-name=grpo-math-neuron
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

set -ex
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1

MODEL="${1:-Qwen/Qwen3-8B}"
CORES_PER_NODE=${CORES_PER_NODE:-64}          # trn2.48xlarge at LNC=2
VLLM_TP=${VLLM_TP:-8}
TRAINER_IMAGE=${TRAINER_IMAGE:-./grpo-math-neuron.sqsh}
VLLM_IMAGE=${VLLM_IMAGE:-./vllm-neuron-0.29.sqsh}
export NEURON_WEIGHT_SYNC_DIR=${NEURON_WEIGHT_SYNC_DIR:-/fsx/${USER}/grpo-math/weight_sync}
mkdir -p "$NEURON_WEIGHT_SYNC_DIR"

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
TRAIN_NODES_NUM=$((SLURM_NNODES - 1))
TRAIN_NODES="${NODELIST[@]:0:$TRAIN_NODES_NUM}"
VLLM_NODE="${NODELIST[$TRAIN_NODES_NUM]}"
head_node_ip=${NODELIST[0]}

# vLLM on AWS Neuron with the in-place weight reload extension (exposed through /collective_rpc).
srun -l --container-image "$VLLM_IMAGE" --container-mounts=/fsx:/fsx,$PWD/neuron:/opt/ext,$HF_HOME:$HF_HOME \
    --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE}" --output=vllm-%j.out \
    --export=ALL,VLLM_SERVER_DEV_MODE=1,NEURON_SKIP_EFA_AFFINITY=1,PYTHONPATH=/opt/ext \
    vllm serve "$MODEL" --host 0.0.0.0 --port 8000 --tensor-parallel-size $VLLM_TP \
      --max-model-len 1024 --max-num-seqs 64 --no-enable-prefix-caching \
      --worker-extension-cls neuron_weight_reload.NeuronWeightReload &

until curl -sf "http://${VLLM_NODE}:8000/health" >/dev/null; do sleep 30; done

LAUNCHER="accelerate launch --config_file /opt/grpo-math/neuron/fsdp2.yaml \
    --num_processes $((TRAIN_NODES_NUM * CORES_PER_NODE)) --num_machines ${TRAIN_NODES_NUM} \
    --rdzv_backend c10d --main_process_ip $head_node_ip --main_process_port 29500 --machine_rank \$SLURM_NODEID"
PREP="python3 -c \"import sys; sys.path.insert(0, '/opt/grpo-math/neuron'); from huggingface_hub import snapshot_download; \
import trl_neuron_weight_sync as s; s.prepare_sync_dir(snapshot_download('$MODEL'))\""

srun -l --container-image "$TRAINER_IMAGE" --container-mounts=/fsx:/fsx,$HF_HOME:$HF_HOME \
    --nodes=$TRAIN_NODES_NUM --ntasks=$TRAIN_NODES_NUM --nodelist="${TRAIN_NODES}" --output=grpo-%j.out \
    bash -c "$PREP && $LAUNCHER /opt/grpo-math/neuron/train_neuron.py --model $MODEL --vllm_server_host $VLLM_NODE"
