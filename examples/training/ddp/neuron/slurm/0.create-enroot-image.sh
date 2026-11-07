#!/usr/bin/env bash
set -ex

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Build the Trainium image from examples/training/ddp and import it for Pyxis.
cd "$(dirname "$0")"
rm -f pytorch-neuron.sqsh
docker build -t pytorch-ddp-neuron -f ../Dockerfile ../..
enroot import -o pytorch-neuron.sqsh dockerd://pytorch-ddp-neuron:latest
