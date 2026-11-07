# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# AWS Trainium (trn2) image for the FSDP example: Neuron SDK 2.33, PyTorch Native (torch 2.14 + torch-neuronx).
# The base image pins torch through PIP_CONSTRAINT, so extras cannot replace the Neuron build of torch.
FROM public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04

# transformers without dependencies (it would otherwise resolve torch); then its pure-Python dependencies.
RUN pip install --no-cache-dir --no-deps transformers==5.22.1 \
 && pip install --no-cache-dir "huggingface_hub>=1.0,<2" regex typer "tokenizers>=0.22,<0.24" \
      safetensors datasets==5.1.0 sentencepiece

ENV FSDP_DEVICE=neuron
# Not /workspace: the base image keeps the editable torch-neuronx source tree in /workspace/torch_neuronx,
# and running from /workspace would shadow the installed package with that directory.
COPY src/ /opt/fsdp/
WORKDIR /opt/fsdp
