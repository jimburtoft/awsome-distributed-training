# PyTorch DDP <!-- omit in toc -->

Isolated environments are crucial for reproducible machine learning because they encapsulate specific software versions and dependencies, ensuring models are consistently retrainable, shareable, and deployable without compatibility issues.

Python [venv](https://docs.python.org/3/library/venv.html) creates lightweight virtual environments to isolate project dependencies, ensuring reproducibility without conflicts between different projects. [Docker](https://www.docker.com/), a containerization platform, packages applications and their dependencies into containers, ensuring they run seamlessly across any Linux server by providing OS-level virtualization and encapsulating the entire runtime environment.

This example showcases [PyTorch DDP](https://pytorch.org/tutorials/beginner/ddp_series_theory.html) environment setup utilizing these approaches for efficient environment management. The implementation supports both CPU and GPU computation:

- **CPU Training**: Uses the GLOO backend for distributed training on CPU nodes
- **GPU Training**: Automatically switches to NCCL backend when GPUs are available, providing optimized multi-GPU training
- **AWS Trainium Training**: With `DDP_DEVICE=neuron`, uses the Neuron backend and NeuronCores through PyTorch Native (TorchNeuron). See [AWS Trainium](#aws-trainium)

## Training

### Basic Usage

To run the training with GPUs, use `torchrun` with the appropriate number of GPUs:

```bash
torchrun --nproc_per_node=N ddp.py --total_epochs=10 --save_every=1 --batch_size=32
```

where N is the number of GPUs you want to use.

## MLFlow Integration

This implementation includes [MLFlow](https://mlflow.org/) integration for experiment tracking and model management. MLFlow helps you track metrics, parameters, and artifacts during training, making it easier to compare different runs and manage model versions.

### Setup

1. Install MLFlow:

```bash
pip install mlflow
```

1. Start the MLFlow tracking server:

```bash
mlflow ui
```

### Usage

To enable MLFlow logging, add the `--use_mlflow` flag when running the training script:

```bash
torchrun --nproc_per_node=N ddp.py --total_epochs=10 --save_every=1 --batch_size=32 --use_mlflow
```

By default, MLFlow will log to `file://$HOME/mlruns`. To use a different tracking server, specify the `--tracking_uri`:

```bash
torchrun --nproc_per_node=N ddp.py --total_epochs=10 --save_every=1 --batch_size=32 --use_mlflow --tracking_uri=http://localhost:5000
```

### What's Tracked

MLFlow will track:

- Training metrics (loss per epoch)
- Model hyperparameters
- Model checkpoints
- Training configuration

### Viewing Results

1. Open your browser and navigate to `http://localhost:5000` (or your specified tracking URI)

The MLFlow UI provides:

- Experiment comparison
- Metric visualization
- Parameter tracking
- Model artifact management
- Run history

## AWS Trainium

The same `ddp.py` runs on AWS Trainium (trn2) with [PyTorch Native](https://awsdocs-neuron.readthedocs-hosted.com/) (Neuron SDK 2.33, torch 2.14 + torch-neuronx). Set `DDP_DEVICE=neuron` to select the `neuron` process-group backend and one NeuronCore per rank. GPU and CPU runs are unchanged.

The [neuron](neuron) directory contains:

| File | Purpose |
|---|---|
| [`neuron/Dockerfile`](neuron/Dockerfile) | Image built on the public Neuron SDK 2.33 PyTorch Native image |
| [`neuron/slurm/0.create-enroot-image.sh`](neuron/slurm/0.create-enroot-image.sh) | Builds the image and imports it for Pyxis |
| [`neuron/slurm/1.container-train.sbatch`](neuron/slurm/1.container-train.sbatch) | Multi-node Slurm job (SageMaker HyperPod or ParallelCluster) |
| [`neuron/kubernetes/ddp-neuron.yaml-template`](neuron/kubernetes/ddp-neuron.yaml-template) | PyTorchJob that requests Neuron devices and EFA |

Run one process per logical NeuronCore: 4 on `trn2.3xlarge` and 64 on `trn2.48xlarge` (at the default logical NeuronCore size of 2).

```bash
# Slurm (from examples/training/ddp/neuron/slurm)
./0.create-enroot-image.sh
NPROC_PER_NODE=64 sbatch --nodes 2 1.container-train.sbatch
```

```bash
# Kubernetes (from examples/training/ddp/neuron/kubernetes)
export IMAGE_URI=<your ECR repository>/ddp-neuron:latest   # built from neuron/Dockerfile
export INSTANCE_TYPE=ml.trn2.48xlarge NUM_NODES=2
export NEURON_DEVICES_PER_NODE=16 NEURON_CORES_PER_NODE=64 EFA_PER_NODE=16
envsubst < ddp-neuron.yaml-template | kubectl apply -f -
```

Single node, without a scheduler:

```bash
docker build -t ddp-neuron -f neuron/Dockerfile .
docker run --rm --privileged --shm-size=8g --entrypoint torchrun ddp-neuron \
  --nproc_per_node=4 /opt/ddp/ddp.py --total_epochs=3 --save_every=1 --batch_size=64
```

## Deployment

We provide guides for both Slurm and Kubernetes. However, please note that the venv example is only compatible with Slurm. For detailed instructions, proceed to the [slurm](slurm) or [kubernetes](kubernetes) subdirectory.
