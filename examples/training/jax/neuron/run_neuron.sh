#!/bin/bash
# Per-process launcher for train_spmd_lm.py: one process per node (all NeuronCores of the node),
# or several processes per node with disjoint cores (NEURON_RT_VISIBLE_CORES).
# Inputs: NNODES/NODE_RANK (or SLURM_*), MASTER_ADDR, CORES_PER_PROC, PROCS_PER_NODE, LOCAL_PROC.
set -euo pipefail
NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
PROCS_PER_NODE=${PROCS_PER_NODE:-1}
LOCAL_PROC=${LOCAL_PROC:-0}
CORES_PER_PROC=${CORES_PER_PROC:-$(neuron-ls --json-output 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(x['nc_count'] for x in d))" 2>/dev/null || echo 4)}
NPROC=$((NNODES * PROCS_PER_NODE))
export JAX_NUM_PROCESSES=$NPROC
export JAX_PROCESS_ID=$((NODE_RANK * PROCS_PER_NODE + LOCAL_PROC))
export JAX_COORDINATOR_ADDRESS=${MASTER_ADDR:-localhost}:${JAX_PORT:-41000}
export NEURON_RT_ROOT_COMM_ID=${MASTER_ADDR:-localhost}:${NEURON_COMM_PORT:-62182}
export NEURON_PJRT_PROCESS_INDEX=$JAX_PROCESS_ID
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(python3 -c "print(','.join(['$CORES_PER_PROC'] * $NPROC))")
if [ "$PROCS_PER_NODE" -gt 1 ]; then
  first=$((LOCAL_PROC * CORES_PER_PROC))
  export NEURON_RT_VISIBLE_CORES=$first-$((first + CORES_PER_PROC - 1))
fi
# EFA (multi-node)
export FI_PROVIDER=${FI_PROVIDER:-efa}
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
exec python3 "$(dirname "$0")/train_spmd_lm.py" "$@"
