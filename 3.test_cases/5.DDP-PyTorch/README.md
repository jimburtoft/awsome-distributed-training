# DDP PyTorch
This is an example of minimal PyTorch DDP code using a toy example to validate if your cluster to is ready to run Pytorch distributed. This test case is based on [this example](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) from PyTorch.

# 0. Preparation

This guide assumes that you have the following:

- A functional Slurm cluster on AWS.
- Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
- A shared directory mounted on `/apps`
- AWS CLI (tested with version `2.7.7`)

It is recommended that you use the templates in the architectures [directory](../../1.architectures). You will also need to export the environment variables below. Please ensure that the region is correct if

```bash
export AWS_REGION=us-east-1
export APPS_PATH=/apps # this is were the squash file (Enroot file) will be stored
export TEST_CASE_PATH=${HOME}/2.PyTorch-DDP-validation # it is assumes that this test case is copied in your home directory
cd ${TEST_CASE_PATH}
```

## 1. Build the Squash file

We will use [AWS Deep learning containers](https://aws.amazon.com/machine-learning/containers/), prebuilt containers for AWS environment (for example PyTorch with EFA support for NCCL operations)

Run the following steps to build the container:

1. Login to AWS ECR with command bellow
   ```bash
   aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com
   ```
2. Pull the container image with the command below
   ```bash
   docker pull 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2
   ```
3. Convert the container image to a squash file via Enroot
   ```bash
   enroot import -o ${APPS_PATH}/ddp-pytorch.sqsh  dockerd://763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2
   ```
   The file will be stored in the `/apps` directory.

## 2. Running the model

It is assumed that your current working directory is `TARGET_PATH` and contains the files `0.run.sbatch` and `1.run.py`.

1. Submit a the PyTorch DDP example as the job using the command below. It will return a job ID:
   ```bash
   sbatch 0.run.sbatch
   ```
2. An output file is generated in your current directory and will contain a string with the job ID

The most important is that we can inspect ranking of our data parallel training, which is shown with `LOCAL_RANK, RANK, GROUP_RANK, LOCAL_WORLD_SIZE, WORLD_SIZE` variables. See [torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html#environment-variables) for definition of these variables.



## 3. Authors / Reviewers

- [A] Uros Lipovsek - lipovsek@
- [R] Pierre-Yves Aquilanti - pierreya@
