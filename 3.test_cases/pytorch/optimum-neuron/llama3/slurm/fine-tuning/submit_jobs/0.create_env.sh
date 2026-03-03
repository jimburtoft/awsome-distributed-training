#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=create_env
#SBATCH -o /fsx/ubuntu/peft_ft/logs/0_create_env.out

set -ex

sudo apt-get update
sudo apt-get install -y python3.10-venv git

srun sudo apt-get update && sudo apt-get install -y python3.10-venv git

python3.10 -m venv /fsx/ubuntu/peft_ft/env_llama3_8B_peft
source /fsx/ubuntu/peft_ft/env_llama3_8B_peft/bin/activate
pip install -U pip

python3 -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"

python3 -m pip install --upgrade neuronx-cc==2.* torch-neuronx==2.8.* torchvision
python3 -m pip install --upgrade neuronx-distributed

python3 -m pip install 'optimum-neuron[training]==0.4.5'
