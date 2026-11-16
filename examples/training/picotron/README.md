# Picotron Test Cases

This test case demonstrates distributed training of [Picotron](https://github.com/huggingface/picotron), a distributed training framework for education and research experimentation. Picotron is designed for training speed, scalability, and memory efficiency.

## Build Environment

The provided Dockerfile (`picotron.Dockerfile`) will set up the environment with all required dependencies:

```bash
docker build -t picotron -f picotron.Dockerfile .
```

First, create a Hugging Face account to retrieve a [token](https://huggingface.co/settings/tokens.). Log in to your account and create an access token from Hugging Face Tokens.

Save the token onto the head node and download the Llama model:

## Get huggingface token

```bash
huggingface-cli login
```

You will be prompted to input the token. Paste the token and answer `n` when asked to add the token as a git credential.

```

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) n
Token is valid (permission: read).
Your token has been saved to /fsx/ubuntu/.cache/huggingface/token
Login successful
```

Then use the saved token `${HF_TOKEN}` to create configuration.

## Next Step

Each subdirectory contains example configurations for training different types of models:

- `SmolLM-1.7B`: Training a 1.7B parameter LLM model on CPU/GPU using 3D parallelism.

## AWS Trainium

picotron runs on AWS Trainium (trn2) with [PyTorch Native](https://awsdocs-neuron.readthedocs-hosted.com/) (Neuron SDK 2.33, torch 2.14 + torch-neuronx). The [neuron](neuron) directory contains:

| File | Purpose |
|---|---|
| [`neuron/Dockerfile`](neuron/Dockerfile) | Image on the public Neuron SDK 2.33 PyTorch Native image with picotron at the same commit |
| [`neuron/patches/`](neuron/patches) | Two picotron patches, which also run on GPUs: (1) transformers 5 compatibility, optional flash-attn and triton imports, `DEVICE=neuron`; (2) optional explicit causal mask for SDPA (`SDPA_EXPLICIT_MASK=1`, default off) |
| [`neuron/slurm/train-neuron.sbatch`](neuron/slurm/train-neuron.sbatch) | Slurm (Pyxis) job |

Create a config with `--use_neuron`, which selects the `neuron` backend and the PyTorch attention path. Then build the image and submit:

```bash
docker build -t picotron-neuron -f neuron/Dockerfile .
enroot import -o picotron-neuron.sqsh dockerd://picotron-neuron:latest
# create_config.py imports picotron, so run it in the image (it also downloads the model to HF_HOME)
export HF_HOME=/fsx/${USER}/.cache/huggingface
srun -N1 --container-image=$PWD/picotron-neuron.sqsh --container-mounts=$PWD,$HF_HOME --container-workdir=$PWD \
    --container-env=HF_TOKEN,HF_HOME python3 /opt/picotron/create_config.py --use_neuron --out_dir conf \
    --exp_name smollm-neuron --model_name HuggingFaceTB/SmolLM-1.7B --dp 16 --tp 4 --seq_len 1024 --mbs 2 --hf_token ${HF_TOKEN}
NPROC_PER_NODE=64 sbatch neuron/slurm/train-neuron.sbatch
```

Data and tensor parallelism work on a single instance. Pipeline and context parallelism send tensors point to point. On Trainium that goes over EFA, so it needs an instance whose Neuron devices have EFA attached (`trn2.48xlarge`); it does not run on `trn2.3xlarge`.

On Trainium1 (trn1), which is included for testing only, set `SDPA_EXPLICIT_MASK=1` along with the settings in [PyTorch Native on AWS Trainium1](../../../docs/neuron-trn1-pytorch-native.md). Tested on HyperPod 2x `ml.trn1n.32xlarge` with SmolLM-1.7B: DP=8 x TP=8 (64 ranks) trained from loss 10.94 to 6.01 in 20 steps. DP=4 x TP=16 (TP groups spanning devices) failed to set up the collectives.
