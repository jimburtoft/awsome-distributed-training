# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# AWS Trainium (trn2) image for the PyTorch DDP example: Neuron SDK 2.33, PyTorch Native (torch 2.14 + torch-neuronx).
# The base image pins torch through PIP_CONSTRAINT, so extras cannot replace the Neuron build of torch.
FROM public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04

# torchvision without dependencies, so it does not pull a CUDA build of torch.
RUN pip install --no-cache-dir --no-deps torchvision==0.29.0 \
 && pip install --no-cache-dir mlflow-skinny==3.17.2 sagemaker-mlflow==0.1.1

ENV DDP_DEVICE=neuron
# Not /workspace: the base image keeps the editable torch-neuronx source tree in /workspace/torch_neuronx,
# and running from /workspace would shadow the installed package with that directory.
COPY ddp.py /opt/ddp/ddp.py
