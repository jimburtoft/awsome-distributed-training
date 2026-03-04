#!/bin/bash

export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export IMAGE=llama3_trn
export TAG=:latest
export IMAGE_URI=${REGISTRY}${IMAGE}${TAG}

export JOB_NAME=trn-llama3-training
export NUM_NODES=1
export INSTANCE_TYPE=ml.trn1.32xlarge  # Options: ml.trn1.32xlarge, ml.trn1n.32xlarge, ml.trn2.48xlarge
export FI_PROVIDER=efa

export FSX_CLAIM=fsx-claim # Change this according to the pvc created.

# Tokenize_data configs
export HF_ACCESS_TOKEN=hf_xxxxxx
export TOKENIZED_DATA_PATH=/fsx/tokenized_data
export DATASET_NAME=wikicorpus
export DATASET_CONFIG_NAME=raw_en
export HF_MODEL_NAME=meta-llama/Meta-Llama-3-8B

# Training configs
export NEURON_CACHE_DIR=/fsx/neuron_cache
export CHECKPOINT_DIR=/fsx/checkpoints
export NUM_KEPT_CHECKPOINTS=2
export CHECKPOINT_FREQ=100
export MAX_STEPS=1000
export STEPS_THIS_RUN=100
export BATCH_SIZE=1
export MODEL_PATH=config_8b_llama3

# Derive parallelism settings from instance type.
# NeuronDevices are the physical devices (used for K8s resource requests).
# NeuronCores are the logical cores (used for nproc_per_node and env vars).
# TP_SIZE is the tensor parallelism degree (capped at 32 by Llama 3 8B's num_attention_heads).
case "$INSTANCE_TYPE" in
    ml.trn1.32xlarge|ml.trn1n.32xlarge)
        export EFA_PER_NODE=8
        export NEURON_PER_NODE=16     # 16 NeuronDevices
        export NEURON_CORES=32        # 32 NeuronCores
        export TP_SIZE=32             # TP=32, DP=1 per node
        ;;
    ml.trn2.48xlarge)
        export EFA_PER_NODE=16
        export NEURON_PER_NODE=16     # 16 NeuronDevices
        export NEURON_CORES=64        # 64 NeuronCores (LNC=2 default)
        export TP_SIZE=32             # TP=32, DP=2 per node
        ;;
    *)
        echo "ERROR: Unsupported instance type: $INSTANCE_TYPE"
        echo "Supported: ml.trn1.32xlarge, ml.trn1n.32xlarge, ml.trn2.48xlarge"
        exit 1
        ;;
esac

# Compute gradient accumulation steps based on parallelism
export DP_PER_NODE=$(($NEURON_CORES / $TP_SIZE))
export TOTAL_DP=$(($DP_PER_NODE * $NUM_NODES))
export GBS=1024
export GRAD_ACCUM_USTEPS=$(($GBS / $BATCH_SIZE / $TOTAL_DP))

echo "Instance: $INSTANCE_TYPE -> NeuronCores=$NEURON_CORES, TP=$TP_SIZE, DP/node=$DP_PER_NODE, grad_accum=$GRAD_ACCUM_USTEPS"

cat tokenize_data.yaml-template | envsubst > tokenize_data.yaml
cat llama3_train.yaml-template | envsubst > llama3_train.yaml

echo "Generated YAML files successfully."
