# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# AWS Trainium (trn2) image for the TorchTitan example. The Neuron SDK 2.33 PyTorch Native image already
# ships torchtitan-neuron (TorchTitan v0.2.2 + Neuron recipes) at /workspace/torchtitan_neuron.
FROM public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04

# Small single-instance smoke configs (Neuron recipe settings on the debug model).
COPY neuron_smoke_configs.py /opt/torchtitan/neuron_smoke_configs.py
ENV PYTHONPATH=/opt/torchtitan
# torchtitan resolves relative asset paths (./assets/hf/..., ./tests/assets/...) from the working directory.
WORKDIR /workspace/torchtitan_neuron
