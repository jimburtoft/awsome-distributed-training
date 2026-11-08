# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# AWS Trainium (trn2) image for the GPT-OSS LoRA SFT step: Neuron SDK 2.33, PyTorch Native.
# Build from this example's directory:  docker build -f neuron/Dockerfile -t gpt-oss-neuron .
# The base image pins torch through PIP_CONSTRAINT; torch-dependent packages are installed with --no-deps.
FROM public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04

RUN pip install --no-cache-dir --no-deps transformers==5.22.1 accelerate==1.15.0 peft==0.21.2 trl==1.6.0 \
 && pip install --no-cache-dir "huggingface_hub>=1.0,<2" "tokenizers>=0.22,<0.24" safetensors regex typer \
      datasets==5.1.0 pyyaml

ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Not /workspace: the base image keeps the editable torch-neuronx source tree in /workspace/torch_neuronx.
COPY artifacts/src/ /opt/gpt-oss/
ENV PYTHONPATH=/opt/gpt-oss
WORKDIR /opt/gpt-oss
