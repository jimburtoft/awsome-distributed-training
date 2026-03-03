#!/bin/bash

###########################
###### User Variables #####
###########################

GPUS_PER_NODE=32
if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    MAX_STEPS=10
    MAYBE_COMPILE="neuron_parallel_compile"
    OUTPUT_DIR="/fsx/ubuntu/peft_ft/compile"
else
    MAX_STEPS=-1
    OUTPUT_DIR="/fsx/ubuntu/peft_ft/model_checkpoints"
fi

###########################
## Environment Variables ##
###########################

CACHE_DIR='/fsx/ubuntu/peft_ft/cache/neuron_compile_cache/llama3-8B'
mkdir -p $CACHE_DIR
export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training --enable-saturate-infinity --cache_dir=$CACHE_DIR"
export OMP_NUM_THREADS=1
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=5
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export MALLOC_ARENA_MAX=70
export FI_PROVIDER="efa"

###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
)

export TRAIN_SCRIPT=/fsx/ubuntu/awsome-distributed-training/3.test_cases/pytorch/optimum-neuron/llama3/src/train.py

############################
##### Training Params ######
############################

# Script-specific arguments (ScriptArguments dataclass)
# NeuronTrainingArguments are passed as standard HuggingFace training args
declare -a TRAINING_ARGS=(
    --model_id "/fsx/ubuntu/peft_ft/model_artifacts/llama3-8B" \
    --dataset "databricks/databricks-dolly-15k" \
    --max_seq_length 2048 \
    --model_final_path "/fsx/ubuntu/peft_ft/model_checkpoints/final" \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --learning_rate 2e-05 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --tensor_parallel_size 8 \
    --logging_steps 1 \
    --save_steps 400 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir
)

source /fsx/ubuntu/peft_ft/env_llama3_8B_peft/bin/activate

$MAYBE_COMPILE torchrun "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"
