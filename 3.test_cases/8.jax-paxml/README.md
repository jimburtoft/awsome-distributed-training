# Jax Paxml
[Pax](https://github.com/google/paxml) is a Jax based framework for machine learning experiments. We will run benchmark with synthetic data.


# 0. Preparation

This guide assumes that you have the following:

- A functional Slurm cluster on AWS.
- Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
- A shared directory mounted on `/apps`

It is recommended that you use the templates in the architectures [directory](../../1.architectures)


## 1. Build the Squash file
1. Build the container image with the command below
   ```bash
   docker build -t jax-paxml -f 0.jax-paxml.Dockerfile .
   ```
2. Convert the container image to a squash file via Enroot
   ```bash
   enroot import -o /apps/jax-paxml.sqsh  dockerd://jax-paxml:latest
   ```
   The file will be stored in the `/apps` directory.


## 2. Running a paxml test

Ensure that the submission file `1.jax-paxml.run-script` has been copied to `/apps/jax-paxml` and given execution privileges (`sudo chmod +x 1.jax-paxml.run-script`) and `2.jax-paxml.run.sbatch` has been copied to `/apps` on your cluster and your shell points to the directory where this file is present. Run the Pax benchmark on 1 node (4 GPUs) as follows:

```bash
sbatch --export=NONE 1.jax-paxml.run-script
```

The command will return a job ID. If you stay in the same directory where `sbatch` was called, assuming that your job is executing (or has executed) you will find a output file for your job names `slurm_<ID>.out` where ID corresponds to your job ID.

We add `--export=NONE` parameter to make sure any conflicting environment variable from AMI is not exported to container.

> **Note**: the number of nodes used for the job is defined via the command line with the option and argument `-N 2`. An alternative is to set it in the `sbatch` file as the directive `#SBATCH -N 2`.

> **Note**: In order to increase number of GPUs (for example to 16 with 2 nodes, 8 GPUs per node) change `ICI_MESH_SHAPE` (for example to `[q, 16, 1]`) in your experiment (for example for `LmCloudSpmd2B` change [here](https://github.com/google/paxml/blob/8210a5f74be0f0fb84d195947462ed65b994b5fa/paxml/tasks/lm/params/lm_cloud.py#L178)). You can accomplish this for example with git patch or defining middle number in `ICI_MESH_SHAPE` as environmental variable an then either calculate or set it withing sbatch script, or with [sbatch CLI](https://slurm.schedmd.com/sbatch.html#:~:text=Example%3A%20%2D%2Dexport,ARG1%3Dtest.).