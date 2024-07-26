# MyoPS2024-UMamba

## Installation


Requirements: `Ubuntu 22.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n umamba python=3.11 -y` and `conda activate umamba `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.4.0: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. Download code: `git clone https://github.com/gaojh135/MyoPS2024-UMamba.git`
5. `cd U-Mamba/umamba` and run `pip install -e .`

sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```