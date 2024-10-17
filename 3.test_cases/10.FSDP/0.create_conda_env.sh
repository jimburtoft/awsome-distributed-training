#!/usr/bin/env bash
set -ex

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -f -p ./miniconda3

source ./miniconda3/bin/activate

conda create -y -p ./pt_fsdp python=3.11

source activate ./pt_fsdp/

conda install -y "pytorch=2.3.*" torchvision torchaudio transformers datasets fsspec=2023.9.2 pytorch-cuda=12.1 --override-channels -c pytorch -c nvidia -c conda-forge

# Install SMDDP from local wheel file
pip install -I ./smdistributed_dataparallel-x.y.z-linux_x86_64.whl

# Create checkpoint dir
mkdir checkpoints
