# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# AWS Trainium (trn2) image for picotron: Neuron SDK 2.33, PyTorch Native (torch 2.14, Python 3.14).
# The base image pins torch through PIP_CONSTRAINT; torch-dependent packages are installed with --no-deps.
FROM public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04

RUN pip install --no-cache-dir --no-deps transformers==5.22.1 \
 && pip install --no-cache-dir "huggingface_hub>=1.0,<2" "tokenizers>=0.22,<0.24" safetensors regex typer \
      datasets==5.1.0 wandb numpy

# picotron at the same commit as the GPU image, with one patch: transformers 5 compatibility
# (batch_encode_plus, rope_parameters), optional flash-attn/triton imports, and DEVICE=neuron support.
COPY neuron/patches/ /tmp/patches/
RUN git clone https://github.com/huggingface/picotron /opt/picotron \
 && cd /opt/picotron && git checkout 59714b1bf255 \
 && git -c user.name=build -c user.email=build@localhost am /tmp/patches/*.patch \
 && pip install --no-cache-dir --no-deps -e .

# Not /workspace: the base image keeps the editable torch-neuronx source tree in /workspace/torch_neuronx.
WORKDIR /opt/picotron
COPY train.py create_config.py ./
COPY template/ ./template/
