#!/usr/bin/env bash
set -ex

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Download the Llama 3.1 tokenizer (gated model: request access on Hugging Face first).
: "${HF_TOKEN:?export HF_TOKEN with access to meta-llama/Llama-3.1-8B}"
cd "$(dirname "$0")"
mkdir -p assets
srun --nodes=1 --ntasks=1 --container-image $(pwd)/torchtitan-neuron.sqsh \
    --container-mounts $(pwd)/assets:/assets --container-env HF_TOKEN \
    python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --local_dir /assets
