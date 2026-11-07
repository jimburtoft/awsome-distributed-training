#!/usr/bin/env bash
set -ex

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Build the Trainium image from examples/training/fsdp and import it for Pyxis.
cd "$(dirname "$0")"
rm -f pytorch-fsdp-neuron.sqsh
docker build -t pytorch-fsdp-neuron -f ../Dockerfile ../..
enroot import -o pytorch-fsdp-neuron.sqsh dockerd://pytorch-fsdp-neuron:latest
