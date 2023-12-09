# Test S3 PyTorch Connector with S3 standard and S3 express

https://github.com/awslabs/s3-connector-for-pytorch/tree/main


## 0. You can export the following environment variables:

```bash
export PYTHON_VERSION=3.10
# We are using Python version 3.10 in this work. For a different Python version select the right Miniconda file from https://repo.anaconda.com/miniconda/
export MINICONDA_INSTALLER=Miniconda3-py310_23.5.2-0-Linux-x86_64
export CUDA_VERSION=12.1
export MOSAICML_VERSION=0.15.0
export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu121
```

## 1. Create conda environment

```bash
# install in the shared directory so compute nodes can source the environment:
cd /apps

# Get the appropriate Miniconda_version from https://repo.anaconda.com/miniconda/
wget -O miniconda.sh "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}.sh" \
    && bash miniconda.sh -b -p ./.conda \
    &&  ./.conda/bin/conda init bash

# Detect the OS based on /etc/os-release
os=$(grep '^ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"')

if [[ "$os" == "amzn" ]]; then
  source /home/ec2-user/.bashrc
elif [[ "$os" == "ubuntu" ]]; then
  source /home/ubuntu/.bashrc
else
  echo "Unknown OS: $os"
fi

conda create -n pt-nightlies python=${PYTHON_VERSION}

conda activate pt-nightlies

# Install PyTorch Nightly distribution with specified Cuda version
pip3 install --pre torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}

# Install s3torchconnector
pip3 install s3torchconnector
pip3 install webdataset
```

