# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# AWS Trainium (trn2) image for the bf16 LoRA + FSDP2 alternative to the QLoRA example.
# Neuron SDK 2.33 PyTorch Native; the base image pins torch through PIP_CONSTRAINT.
FROM public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04

RUN pip install --no-cache-dir --no-deps transformers==5.22.1 accelerate==1.15.0 peft==0.21.2 \
 && pip install --no-cache-dir "huggingface_hub>=1.0,<2" "tokenizers>=0.22,<0.24" safetensors regex typer \
      datasets==5.1.0 pyyaml tensorboard

# Not /workspace: the base image keeps the editable torch-neuronx source tree in /workspace/torch_neuronx,
# and running from /workspace would shadow the installed package with that directory.
COPY src/ /opt/qlora/src/
COPY configs/ /opt/qlora/configs/
COPY neuron/ /opt/qlora/neuron/
WORKDIR /opt/qlora
