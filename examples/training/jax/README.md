# JAX container for Amazon EC2 GPU accelerated Instances

Ths directory contains a sample Dockerfile `jax_paxml.Dockerfile` to run [JAX](https://github.com/google/jax) and [Paxml](https://github.com/google/paxml) on AWS.

## Container description

In principle, the reference `Dockerfile` does the following:

- Provide JAX built for NVIDIA CUDA devices, by using a recent NVIDIA CUDA image as the
  parent image.
- Remove unneccessary networking packages that might conflict with AWS technologies.
- Install EFA user-space libraries. It's important to avoid building the kernel drivers during
  `docker build`, and skip the self-tests, as both of these steps fail are expected to fail when run
  during container build.
- Install NCCL recommended version.
- Install [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) to get NCCL to utilize EFA.
- Install JAX.
- Install Paxml.
- Install Praxis.

## Build the container

Build the jax container as follow

```bash
# Build a container image
DOCKER_BUILDKIT=1 docker build --progress=plain -f jax_paxml.Dockerfile -t paxml:jax-0.4.18-1.2.0 .

# Verify the image has been built
docker images
```

Convert container to enroot format

```bash
# Convert to enroot format. Attempt to remove an existing .sqsh, otherwise enroot refuses to
# run when the output .sqsh file already exists.
rm /fsx/paxml_jax-0.4.18-1.2.0.sqsh ; enroot import -o /fsx/paxml_jax-0.4.18-1.2.0.sqsh dockerd://paxml:jax-0.4.18-1.2.0
```

## Run

Once the container converted to the enroot format, you can run on a **Slurm** cluster the lm_clouds example of Pax.
The following command submit a job to the **Slurm** cluster to train a 2B parameters transformer based SPMD language model on synthetic data.

```bash
sbatch jax.sbatch
```

## AWS Trainium (JAX NeuronX)

Paxml and Praxis have no AWS Neuron backend. [`neuron/`](neuron/) holds a plain-JAX counterpart of the `LmCloudSpmd2B` run above:

- a decoder-only transformer LM trained on synthetic tokens;
- SPMD-sharded with `jax.sharding` over a `(data, model)` mesh: data parallel × Megatron-style tensor parallel;
- AdamW and gradient clipping.

It needs only `jax`, and runs in the public JAX NeuronX image `public.ecr.aws/neuron/jax-training-neuronx:0.10.0-neuronx-py314-sdk2.33.0-ubuntu26.04` (Neuron SDK 2.33, JAX 0.10.0).

| File | Purpose |
|------|---------|
| `neuron/train_spmd_lm.py` | Model, sharding, optimizer, and training loop. `--config tiny/125m/1b/2b`, `--model-parallel N` |
| `neuron/run_neuron.sh` | Per-process launcher: sets the `jax.distributed` and `NEURON_PJRT_*` variables, and EFA |
| `neuron/jax.sbatch` | Slurm job, one task per trn2 node |

Single node (all NeuronCores of the instance):

```bash
docker run --rm --device=/dev/neuron0 --privileged -v "$(pwd)/neuron":/opt/jaxlm -w /opt/jaxlm \
  public.ecr.aws/neuron/jax-training-neuronx:0.10.0-neuronx-py314-sdk2.33.0-ubuntu26.04 \
  python3 train_spmd_lm.py --config 125m --per-device-batch 8 --seq 1024 --steps 300
```

On trn2.48xlarge, pass every `/dev/neuron*` device (16 of them).

Slurm (Pyxis). Build the image from `neuron/Dockerfile`, which clears a multi-line label in the public image that otherwise stops enroot from starting the container:

```bash
docker build -t jax-neuron -f neuron/Dockerfile .
enroot import -o /fsx/jax-neuron.sqsh dockerd://jax-neuron:latest
cd neuron && sbatch --nodes=2 jax.sbatch   # CONFIG=1b MODEL_PARALLEL=4 by default
```

Measured on one trn2.3xlarge (4 NeuronCores, LNC=2), sequence 1024, vocab 32768:

| Config | Mesh (data × model) | Global batch | Steady-state tokens/s | Loss (step 0 → last) |
|--------|---------------------|--------------|-----------------------|----------------------|
| 125M | 4 × 1 | 32 | 113,548 | 10.55 → 3.10 (300 steps) |
| 125M | 2 × 2 | 8 | 58,946 | 10.56 → 9.98 (150 steps) |
| 1B (872M) | 1 × 4 | 4 | 20,387 | 10.79 → 9.49 (300 steps) |

Equal global batches give the same loss curve for every mesh shape (data 4, data 2 × model 2, model 4). That checks the sharding.
The first step includes compilation, which takes 1.5-9 minutes.

Known issues in JAX NeuronX 0.10 (SDK 2.33) that the script works around:

- **Tied embedding sharded on the vocabulary axis.** `grad` of `CE(E[tokens] @ E.T)` with `E` sharded `P("model", None)` is wrong (relative error ~20). The same program on CPU devices is correct, and the forward loss is unaffected, so training silently stalls. The script shards the embedding on the hidden dimension, `P(None, "model")`.
- **`take_along_axis` cross-entropy backward.** With the 1B config sharded on `model`, the backward fails with `NRT_EXEC_OOB` on some batches, even though all token ids are in range. The script uses a one-hot cross-entropy instead. `CE_IMPL=take` restores the original.

The same image runs on Trainium1 (trn1), which is included for testing only. On HyperPod 2x `ml.trn1n.32xlarge` (2 processes, 64 NeuronCores), the 125M config trained from loss 10.55 to 7.03 in 40 steps at about 1.0M tokens/s. The 1B config with model parallelism 8 went from 10.81 to 7.75. trn1 has 16 GiB of HBM per NeuronCore, so use about half the per-device batch of trn2.

