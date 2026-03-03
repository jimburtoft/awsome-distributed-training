## PEFT Fine Tuning of Llama 3 on Amazon EKS with AWS Trainium

This example demonstrates how to perform supervised fine tuning for Meta Llama 3.1 using Parameter-Efficient Fine Tuning (PEFT) on AWS Trainium with EKS. It uses [Hugging Face Optimum Neuron](https://huggingface.co/docs/optimum-neuron) to apply Low-Rank Adaptation (LoRA) for distributed training on Trainium.

The training script is adapted from the [official upstream example](https://github.com/huggingface/optimum-neuron/blob/main/examples/training/llama/finetune_llama.py) and uses:

- **`NeuronModelForCausalLM`** for tensor-parallel model loading
- **`NeuronSFTTrainer`** with **LoRA** (PEFT) for parameter-efficient fine tuning
- **Flash Attention 2** for memory-efficient attention
- **Chat template formatting** with sequence packing
- The [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset

### Software Versions

| Package | Version |
|---------|---------|
| optimum-neuron | 0.4.5 |
| trl | 0.24.0 |
| peft | 0.17.0 |
| transformers | ~4.57 |
| torch | 2.8.0 |
| neuronx-distributed | 0.17.x |

### Solution Overview

This solution uses:
- AWS Trainium chips for deep learning acceleration
- Hugging Face Optimum Neuron for integrating Trainium with existing models and tools
- LoRA for parameter-efficient fine tuning
- Kubeflow PyTorchJob for distributed training orchestration

## 0. Prerequisites

### 0.1. EKS Cluster

Before running this training, you'll need an Amazon EKS or SageMaker HyperPod EKS cluster with at least 1 trn1.32xlarge or trn1n.32xlarge node. Instructions can be found in [1.architectures](../../1.architectures), the [aws-do-eks](https://bit.ly/do-eks) project, or the [eks-blueprints](https://github.com/aws-ia/terraform-aws-eks-blueprints) project.

### 0.2. Setup Persistent Volume Claim (PVC) for FSx

Set up a PVC for FSx to store model artifacts and training checkpoints. Follow the instructions [here](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/01-cluster/06-fsx-for-lustre) to set up the FSx CSI Driver and PVC.

### 0.3. Hugging Face Access Token

Since Llama 3 is a gated model, register on [Hugging Face](https://huggingface.co) and obtain an [access token](https://huggingface.co/docs/hub/en/security-tokens).

## 1. Build and Push Docker Image

### Pull the base image

Login to ECR and pull the `pytorch-training-neuronx` base image:

```sh
region=us-east-1
dlc_account_id=763104351884
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $dlc_account_id.dkr.ecr.$region.amazonaws.com

docker pull ${dlc_account_id}.dkr.ecr.${region}.amazonaws.com/huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04
```

### Build Docker Image

Build the Docker image from the Dockerfile, which installs `optimum-neuron[training]==0.4.5` on top of the base DLC image:

```sh
# Navigate to the llama3 source directory
cd ../../
```

```sh
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export IMAGE=peft-optimum-neuron
export TAG=:latest
docker build -t ${REGISTRY}${IMAGE}${TAG} -f kubernetes/fine-tuning/Dockerfile .
```

### Push to ECR

```sh
# Create registry if needed
export REGISTRY_COUNT=$(aws ecr describe-repositories | grep \"${IMAGE}\" | wc -l)
if [ "${REGISTRY_COUNT//[!0-9]/}" == "0" ]; then
    echo "Creating repository ${REGISTRY}${IMAGE} ..."
    aws ecr create-repository --repository-name ${IMAGE}
else
    echo "Repository ${REGISTRY}${IMAGE} already exists"
fi

# Login to registry
echo "Logging in to $REGISTRY ..."
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY

# Push image to registry
docker image push ${REGISTRY}${IMAGE}${TAG}
```

## 2. Generate Job Spec Files

Edit the `generate-jobspec.sh` script with your environment settings. Key configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_ID` | Hugging Face model ID | `meta-llama/Llama-3.1-8B-Instruct` |
| `MODEL_OUTPUT_PATH` | FSx path for model storage | `/fsx/peft_ft/model_artifacts/llama3-8B` |
| `HF_TOKEN` | Your Hugging Face token | (must be set) |
| `MAX_SEQ_LENGTH` | Max sequence length (multiple of 2048) | `2048` |
| `TP_SIZE` | Tensor parallelism degree | `8` |
| `MAX_TRAINING_STEPS` | Max training steps (-1 for full epoch) | `-1` |
| `CHECKPOINT_DIR` | FSx path for checkpoints | `/fsx/peft_ft/model_checkpoints` |

```bash
./generate-jobspec.sh
```

This creates the following YAML files from templates: `tokenize_data.yaml`, `compile_peft.yaml`, `launch_peft_train.yaml`, `consolidation.yaml`, and `merge_lora.yaml`.

## 3. Download Model

> **Note:** The old tokenize_data step has been replaced with a model download step. The new training script handles tokenization internally via `NeuronSFTTrainer`.

```bash
kubectl apply -f ./tokenize_data.yaml
```

This downloads the Llama 3 model and tokenizer to your FSx volume.

## 4. Compile the Model

```bash
kubectl apply -f ./compile_peft.yaml
```

This pre-compiles the model using `neuron_parallel_compile`, which:
- Extracts computation graphs from a trial run (~10 training steps)
- Performs parallel pre-compilation of these graphs
- Generates NEFF files cached for reuse during training
- Uses the same training script with `--max_steps=10` and `NEURON_EXTRACT_GRAPHS_ONLY=1`

## 5. Train Model

```bash
kubectl apply -f ./launch_peft_train.yaml
```

Training uses:
- **Tensor parallelism degree 8** across all 32 NeuronCores on a trn1.32xlarge
- **Data parallelism degree 4** (32 cores / 8 TP = 4 DP workers)
- **BFloat16** precision
- **LoRA** targeting all linear projections (q, k, v, o, gate, up, down)
- **Sequence packing** for efficient batching
- **Max sequence length 2048** (required minimum for flash attention)

## 6. Consolidate Trained Weights

```bash
kubectl apply -f ./consolidation.yaml
```

During distributed training, model checkpoints are split across tensor parallel devices. The consolidation step combines these shards into a unified `model.safetensors` file using optimum-neuron's built-in consolidation utility.

> **Note:** Update the `MAX_TRAINING_STEPS` in `generate-jobspec.sh` to match the actual final checkpoint step number before running consolidation.

## 7. Merge LoRA Weights

```bash
kubectl apply -f ./merge_lora.yaml
```

This merges the LoRA adapter weights with the base model, producing a final model that can be used for inference without the LoRA configuration.

The resulting merged model combines the base model's knowledge with the task-specific adaptations learned during fine tuning.
