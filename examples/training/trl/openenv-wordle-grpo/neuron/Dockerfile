# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# The Wordle environment client is installed from a plain clone: the Hugging Face Space does not serve the
# partial clone (--filter=blob:none) that `pip install git+...` uses.
# AWS Trainium (trn2) GRPO trainer image for the Wordle example: Neuron SDK 2.33 PyTorch Native + TRL 1.x.
# Generation runs in a separate vLLM on AWS Neuron server (see the README), so vLLM is not installed here.
FROM public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04

RUN pip install --no-cache-dir --no-deps transformers==5.22.1 accelerate==1.15.0 peft==0.21.2 trl==1.17.1 \
 && pip install --no-cache-dir "huggingface_hub>=1.0,<2" "tokenizers>=0.22,<0.24" safetensors regex typer \
      datasets==5.1.0 requests rich "websockets>=15.0,<16.0" "fastapi>=0.115.0" "pydantic>=2.0.0" \
 && pip install --no-cache-dir "openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git@fix-factory-in-ui" \
 && GIT_LFS_SKIP_SMUDGE=1 git clone -q https://huggingface.co/spaces/burtenshaw/wordle /tmp/wordle \
 && git -C /tmp/wordle checkout -q 77bdfedb9144b341db1c37449328709ccbed7da3 \
 && pip install --no-cache-dir --no-deps /tmp/wordle && rm -rf /tmp/wordle

# Not /workspace: the base image keeps the editable torch-neuronx source tree in /workspace/torch_neuronx.
COPY src/ /opt/wordle-grpo/src/
COPY neuron/ /opt/wordle-grpo/neuron/
WORKDIR /opt/wordle-grpo/src
