# DDP PyTorch
[Param benchmark](https://github.com/facebookresearch/param/tree/main) is aa PyTorch benchmark for computation and communication. This guide only addresses communications performances.

# 0. Preparation

This guide assumes that you have the following:

- A functional Slurm cluster on AWS.
- Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
- A shared directory mounted on `/apps`

It is recommended that you use the templates in the architectures [directory](../../1.architectures)


## 1. Build the Squash file

The [AWS Deep learning containers](https://aws.amazon.com/machine-learning/containers/) is used as a base for this project and the Param benchmark is built on top of it following the steps below. It is assumed that you copied the assets (`Dockerfile` and `sbatch` file) to your cluster.

1. Login to AWS ECR with command bellow
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
   ```
2. Build the container image with the command below
   ```bash
   docker build -t param-benchmark -f 0.param-benchmark.Dockerfile .
   ```
3. Convert the container image to a squash file via Enroot
   ```bash
   enroot import -o /apps/param-benchmark.sqsh  dockerd://param-benchmark:latest
   ```
   The file will be stored in the `/apps` directory.


## 2. Running the model

Ensure that the submission file `1.param-benchmark.sbatch` has been copied to your cluster and your shell points to the directory where this file is present. Run the Param benchmark on 2 nodes as follows:

```bash
sbatch -N 2 1.param-benchmark.sbatch
```



The command will return a job ID. If you stay in the same directory where `sbatch` was called, assuming that your job is executing (or has executed) you will find text output files with the standard output and standard error output of your job. These will be named   you find your job output files stdout of the job in file `1.run.sbatch_<job ID>.out` and stderr in `1.run.sbatch_<job ID>.err`.

> **Note**: the number of nodes used for the job is defined via the command line with the option and argument `-N 2`. An alternative is to set it in the `sbatch` file as the directive `#SBATCH -N 2`.


You will see NCCL performance in logs, refer to [NCCL-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) documentation to understand the difference between AlbBW and BusBw. Review [documentation](https://github.com/facebookresearch/param/tree/6236487e8969838822b52298c2a2318f6ac47bbd/train/comms/pt) for other CLI parameters and other benchmarks in the [Param repository repository](https://github.com/facebookresearch/param/tree/6236487e8969838822b52298c2a2318f6ac47bbd) for more communication and computation tests.
