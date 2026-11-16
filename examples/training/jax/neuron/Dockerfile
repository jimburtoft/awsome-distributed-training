# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# JAX NeuronX image for the SPMD LM example (Neuron SDK 2.33, JAX 0.10).
# Build from examples/training/jax:  docker build -f neuron/Dockerfile -t jax-neuron .
FROM public.ecr.aws/neuron/jax-training-neuronx:0.10.0-neuronx-py314-sdk2.33.0-ubuntu26.04
# The base image's description label spans several lines; enroot (Pyxis) copies labels into /etc/rc
# and a multi-line value breaks container start. An empty label avoids it.
LABEL org.opencontainers.image.description=""
COPY neuron/ /opt/jaxlm/
WORKDIR /opt/jaxlm
