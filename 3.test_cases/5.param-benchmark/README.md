# DDP PyTorch
[Param benchmark](https://github.com/facebookresearch/param/tree/main) is PyTorch benchmark for computation and communication performance. We will focus on communication performance in this tutorial, but it can serve as a starting point for computation benchmark which can be run on single node (`torchrun` is not needed) with different benchmark script and changed parameters.

# 0. Preparation

This guide assumes that you have the following:

- A functional Slurm cluster on AWS.
- Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
- A shared directory mounted on `/apps`

It is recommended that you use the templates in the architectures [directory](../../1.architectures)


## 1. Build the Squash file

We will use [AWS Deep learning containers](https://aws.amazon.com/machine-learning/containers/), prebuilt containers for AWS environment (for example PyTorch with EFA support for NCCL operations)
To build the container:

1. Login to AWS ECR with command bellow
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
   ```
2. Build the container image with the command below
   ```bash
   docker build -t param .
   ```
3. Convert the container image to a squash file via Enroot
   ```bash
   enroot import -o /apps/param.sqsh  dockerd://param
   ```
   The file will be stored in the `/apps` directory.


## 2. Running the model

Copy the file `1.run.sbatch` to folder `/apps/param` and then submit a preprocessing jobs with the command below:

```bash
sbatch -N 2 1.run.sbatch
```

Command will return job ID. If you stay in the same directory where you ran sbatch command above you can find stdout of the job in file `1.run.sbatch_<job ID>.out` and stderr in `1.run.sbatch_<job ID>.err`.


You will see NCCL performance in logs, refer to [NCCL-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) documentation to udnerstand the difference between AlbBW and BusBw. Review [documentation](https://github.com/facebookresearch/param/tree/6236487e8969838822b52298c2a2318f6ac47bbd/train/comms/pt) for other CLI parameters and other benchmarks in [param repository](https://github.com/facebookresearch/param/tree/6236487e8969838822b52298c2a2318f6ac47bbd) for more communication and computation tests.