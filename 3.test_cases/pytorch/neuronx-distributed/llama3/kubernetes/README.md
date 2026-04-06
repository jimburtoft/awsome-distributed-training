## Train Llama 3 8B model on Kubernetes

In this section, we showcase how to pre-train Llama3-8B using AWS Trainium instances and the Neuron Distributed library. This example supports both trn1 and trn2 instance types.

**Supported instances:** Set the `INSTANCE_TYPE` variable in `generate-jobspec.sh` to match your cluster. Parallelism settings are derived automatically.

| Instance Type | NeuronCores | TP Degree | DP per Node |
|---------------|-------------|-----------|-------------|
| ml.trn1.32xlarge / ml.trn1n.32xlarge | 32 | 32 | 1 |
| ml.trn2.48xlarge | 64 | 32 | 2 |

> **trn2 status:** The `generate-jobspec.sh` and yaml templates support trn2.48xlarge configuration, but NxD training on trn2 requires Neuron SDK collectives support that is not yet available as of SDK 2.28. The trn1 configuration is fully tested and recommended for production use.

The following optimizations are applied using the Neuron Distributed library:

1. [Tensor Parallelism](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tensor_parallelism_overview.html#tensor-parallelism-overview)

2. [Sequence Parallel](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html#sequence-parallelism)

3. [Selective checkpointing](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html#activation-memory-reduction)

4. [ZeRO-1](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/zero1_gpt2.html#zero1-gpt2-pretraining-tutorial)


## 0. Prerequisites

### 0.1. EKS Cluster 
Before running this training, you'll need to create an Amazon EKS or a SageMaker HyperPod EKS cluster with at least 1 Trainium node (trn1.32xlarge, trn1n.32xlarge, or trn2.48xlarge). Instructions can be found in [1.architectures](../../1.architectures), the [aws-do-eks](https://bit.ly/do-eks) project, or the [eks-blueprints](https://github.com/aws-ia/terraform-aws-eks-blueprints) project.

### 0.2 HF Access token 

Since [llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B) is a gated model users have to register in Huggingface and obtain an HF_Access_Token before running this example.

### 0.3 Setup Persistant Volume Claim(PVC) for fsx 

We need to setup an PVC for FSx to store the tokenized data and training checkpoints. Please follow the link [here](#) to setup FSx CSI Driver and PVC. 

## 1. Setting up environment


### Pull the pytorch-training-neuronx image locally

Login to ECR and pull the `pytorch-training-neuronx` image

```sh
region=us-east-2
dlc_account_id=763104351884
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $dlc_account_id.dkr.ecr.$region.amazonaws.com

docker pull ${dlc_account_id}.dkr.ecr.${region}.amazonaws.com/pytorch-training-neuronx:2.9.0-neuronx-py312-sdk2.28.0-ubuntu24.04
```

### Build Docker Image and push to ECR

We will build docker image using the [Dockerfile](Dockerfile) in this directory.  

```sh
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export IMAGE=llama3_trn
export TAG=:latest
docker build -t ${REGISTRY}${IMAGE}${TAG} .
```

Then push the image to your private registry

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

## Generate Job Spec Files for tokenization and training

The default config in the script launches a 8B Llama 3 model on trn1.32xlarge. To use trn2.48xlarge, change the `INSTANCE_TYPE` variable in the script. Parallelism settings (NeuronCores, TP degree, gradient accumulation) are derived automatically from the instance type.

You will have to update the HF_ACCESS_TOKEN in order for the tokenization to work.

Please edit the `./generate-jobspec.sh` script with your desired environment settings.

```bash
./generate-jobspec.sh
```

## Tokenize Data

```bash
kubectl apply -f ./tokenize_data.yaml
```

## Train Model

```bash
kubectl apply -f ./train_llama3.yaml
```
