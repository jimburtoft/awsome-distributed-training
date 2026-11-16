# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# AWS Trainium (trn2) GRPO trainer image for grpo-math-reasoning: Neuron SDK 2.33 PyTorch Native + TRL 1.x.
# Build from this example's directory:  docker build -f neuron/Dockerfile -t grpo-math-neuron .
# Generation runs in a separate vLLM on AWS Neuron server (public image), so vLLM is not installed here.
# The base image pins torch through PIP_CONSTRAINT; torch-dependent packages are installed with --no-deps.
FROM public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04
# The base image's description label spans several lines; enroot (Pyxis) copies labels into /etc/rc
# and a multi-line value breaks container start. An empty label avoids it.
LABEL org.opencontainers.image.description=""

RUN pip install --no-cache-dir --no-deps transformers==5.22.1 accelerate==1.15.0 peft==0.21.2 trl==1.17.1 \
 && pip install --no-cache-dir "huggingface_hub>=1.0,<2" "tokenizers>=0.22,<0.24" safetensors regex typer \
      datasets==5.1.0 requests rich math-verify==0.8.0

# Not /workspace: the base image keeps the editable torch-neuronx source tree in /workspace/torch_neuronx.
COPY train.py eval.py inference.py /opt/grpo-math/
COPY neuron/ /opt/grpo-math/neuron/
WORKDIR /opt/grpo-math
