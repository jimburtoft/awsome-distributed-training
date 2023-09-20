# DDP PyTorch
This is an example of minimal PyTorch DDP code, usefull for debugging of your infrastructure and some basic validation your cluster is prepared for Pytorch distributed.

# 0. Preparation

This guide assumes that you have the following:

- A functional Slurm cluster on AWS.
- Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
- A shared directory mounted on `/apps`
- AWS CLI (tested with version `2.7.7`)

It is recommended that you use the templates in the architectures [directory](../../1.architectures)


## 1. Build the Squash file

We will use [AWS Deep learning containers](https://aws.amazon.com/machine-learning/containers/), prebuilt containers for AWS environment (for example PyTorch with EFA support for NCCL operations)

Run the following steps to build the container:

1. Login to AWS ECR with command bellow
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
   ```
2. Pull the container image with the command below
   ```bash
   docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2
   ```
3. Convert the container image to a squash file via Enroot
   ```bash
   enroot import -o /apps/ddp-pytorch.sqsh  dockerd://763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2
   ```
   The file will be stored in the `/apps` directory.


## 2. Running the model

Copy files `0.run.sbatch` and `1.run.py` to folder `/apps/ddp-pytorch` and then submit a preprocessing jobs with the command below:

```bash
sbatch 0.run.sbatch
```

The most important is that we can inspect ranking of our data parallel trianing, which is shown with `LOCAL_RANK, RANK, GROUP_RANK, LOCAL_WORLD_SIZE, WORLD_SIZE` variables. See [torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html#environment-variables) for definition of these variables.
