#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

echo "Setting up environment variables"

# Prompt user for WANDB_API_KEY
echo "Please enter your WANDB_API_KEY"
read WANDB_API_KEY

echo "export FSX_PATH=/fsx" > .env
source .env
echo "export IMAGE=torchtitan-torchtune" >> .env
source .env
echo "export APPS_PATH=${FSX_PATH}/apps" >> .env
source .env
echo "export ENROOT_IMAGE=$APPS_PATH/${IMAGE}.sqsh" >> .env
source .env
echo "export MODEL_PATH=$FSX_PATH/models/torchtitan-torchtune" >> .env
source .env
echo "export TEST_CASE_PATH=${FSX_PATH}/awsome-distributed-training/3.test_cases/torchtitan-torchtune/slurm" >> .env
source .env
echo "export HF_HOME=${FSX_PATH}/.cache" >> .env
source .env
echo "export WANDB_CONFIG_DIR=${FSX_PATH}" >> .env
source .env
echo "export WANDB_API_KEY=${WANDB_API_KEY}" >> .env
source .env

echo ".env file created successfully"
echo "Please run 'source .env' to set the environment variables"