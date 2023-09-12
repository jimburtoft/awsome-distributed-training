#! /bin/bash
set -x;
# LmCloudSpmd2B8PLimitSteps
EXP=${1:-LmCloudSpmd2B}
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true"

env

export TPU_TYPE=gpu
export TF_FORCE_GPU_ALLOW_GROWTH=true
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1

PAXML_PREFIX=/opt/paxml


export VOCAB_PATH=${GPT_VOCAB_PATH}
echo ${N_GPUS};

echo "hostname: $(hostname)"
echo "SLURM LAUNCH NODE: ${SLURM_LAUNCH_NODE_IPADDR}"
echo "SLURM NTASKS: ${SLURM_NTASKS}"
echo "SLURM PROCID: ${SLURM_PROCID}"

nvidia-smi -L

echo "Running jax device discovery check.."
python3 -c "import jax; print(jax.devices())"
export PYTHONPATH=${PYTHONPATH}:/praxis:${PAXML_PREFIX}

python3 ${PAXML_PREFIX}/paxml/main.py \
    --job_log_dir=${LOG_DIR} \
    --exp=paxml.tasks.lm.params.lm_cloud.${EXP} \
    --tfds_data_dir=${TFDS_DATA_DIR} \
    --alsologtostderr \
    --multiprocess_gpu \
    --server_addr=${SLURM_LAUNCH_NODE_IPADDR}:12345 \
    --num_hosts=${SLURM_NTASKS} \
    --host_idx=${SLURM_PROCID} \
    --alsologtostderr