#!/usr/bin/env bash
set -ex

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

cd "$(dirname "$0")"
rm -f torchtitan-neuron.sqsh
docker build -t torchtitan-neuron .
enroot import -o torchtitan-neuron.sqsh dockerd://torchtitan-neuron:latest
