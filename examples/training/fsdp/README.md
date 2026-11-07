# Get Started Training Llama 2, Mixtral 8x7B, and Mistral Mathstral with PyTorch FSDP in 5 Minutes

This content provides a quickstart with multinode PyTorch [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) training on Slurm and Kubernetes.
It is designed to be simple with no data preparation or tokenizer to download, and uses Python virtual environment.

## Prerequisites

To run FSDP training, you will need to create a training cluster based on Slurm or Kubermetes with an [Amazon FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html)
You can find instruction how to create a Amazon SageMaker Hyperpod cluster with [Slurm](https://catalog.workshops.aws/sagemaker-hyperpod/en-US), [Kubernetes](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US) or with in [Amazon EKS](../../../architectures).

## FSDP Training

This fold provides examples on how to train with PyTorch FSDP with Slurm or Kubernetes.
You will find instructions for [Slurm](slurm) or [Kubernetes](kubernetes) in the subdirectories. For AWS Trainium, see [AWS Trainium](#aws-trainium).

## AWS Trainium

The same `src/train.py` and model recipes run on AWS Trainium (trn2) with [PyTorch Native](https://awsdocs-neuron.readthedocs-hosted.com/) (Neuron SDK 2.33, torch 2.14 + torch-neuronx). Set `FSDP_DEVICE=neuron` to select the `neuron` process-group backend and one NeuronCore per rank. GPU runs are unchanged.

The [neuron](neuron) directory contains:

| File | Purpose |
|---|---|
| [`neuron/Dockerfile`](neuron/Dockerfile) | Image built on the public Neuron SDK 2.33 PyTorch Native image |
| [`neuron/slurm/0.create-enroot-image.sh`](neuron/slurm/0.create-enroot-image.sh) | Builds the image and imports it for Pyxis |
| [`neuron/slurm/1.llama3-fsdp-neuron.sbatch`](neuron/slurm/1.llama3-fsdp-neuron.sbatch) | Slurm job that reads any recipe in [`models`](models) (`MODEL=llama3_2_1b` by default) |
| [`neuron/kubernetes/llama3_2_1b-fsdp-neuron.yaml`](neuron/kubernetes/llama3_2_1b-fsdp-neuron.yaml), [`neuron/kubernetes/llama3_1_8b-fsdp-neuron.yaml`](neuron/kubernetes/llama3_1_8b-fsdp-neuron.yaml) | PyTorchJobs that request Neuron devices and EFA |

Run one process per logical NeuronCore: 4 on `trn2.3xlarge` and 64 on `trn2.48xlarge` (at the default logical NeuronCore size of 2).

```bash
# Slurm (from examples/training/fsdp/neuron/slurm)
./0.create-enroot-image.sh
MODEL=llama3_1_8b NPROC_PER_NODE=64 sbatch --nodes 2 1.llama3-fsdp-neuron.sbatch
```

```bash
# Kubernetes (from examples/training/fsdp/neuron/kubernetes)
export IMAGE_URI=<your ECR repository>/fsdp-neuron:latest   # built from neuron/Dockerfile
export INSTANCE_TYPE=ml.trn2.48xlarge NUM_NODES=2
export NEURON_DEVICES_PER_NODE=16 NEURON_CORES_PER_NODE=64 EFA_PER_NODE=16
envsubst < llama3_1_8b-fsdp-neuron.yaml | kubectl apply -f -
```

For MFU, pass `--peak_tflops_per_gpu` with the peak of one rank. On trn2 at the default logical NeuronCore size, a rank is one logical core of two physical NeuronCore-v3 cores (79 dense BF16 TFLOPS each), so use `--peak_tflops_per_gpu=158`.
