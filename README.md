# MyoPS2024-UMamba

## Installation


Requirements: `Ubuntu 22.04`, CUDA 11.8

Create a virtual environment: conda create -n umamba python=3.11 -y and conda activate umamba 
Install Pytorch 2.0.1: pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
Install Mamba: pip install causal-conv1d>=1.2.0 and pip install mamba-ssm --no-cache-dir
Download code: git clone https://github.com/bowang-lab/U-Mamba
cd U-Mamba/umamba and run pip install -e .