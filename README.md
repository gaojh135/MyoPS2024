# RGU-Mamba

## Installation


Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n umamba python=3.10 -y` and `conda activate umamba `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. Download code: `git clone https://github.com/bowang-lab/U-Mamba`
5. `cd U-Mamba/umamba` and run `pip install -e .`
6. `pip install PYMIC`

sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```

## Coarse Segmentation

### Data conversion
```bash
python data_conversion_coarse_stage.py
```
### Crop images
```bash
python crop_for_coarse_segment.py
```
### Histogram Matching
```bash
match_for_coarse_segment.py
```

## Model Training

```bash
…………
```

## Fine Segmentation
### Data conversion
```bash
python data_conversion_fine_stage.py
```
### Crop images
```bash
crop_for_fine_segment.py
```
### Histogram Matching
```bash
match_for_fine_segment.py
```





## Model Training

### Data preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Train 2D models

```bash
nnUNetv2_train DATASET_ID 2d FOLD -tr nnUNetTrainerUMambaEnc
```

### Train 3D models

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerUMambaEnc
```

- Find best configuration

```bash
nnUNetv2_find_best_configuration DATASET_ID -c 2d 3d_fullres -tr nnUNetTrainerUMambaEnc
```

- Predict
```bash
nnUNetv2_predict -d DATASET_ID -i INPUT_FOLDER -o OUTPUT_FOLDER_MODEL_1 -f  0 1 2 3 4 -tr nnUNetTrainerUMambaEncNoAMP -c 2d -p nnUNetPlans --save_probabilities

nnUNetv2_predict -d DATASET_ID -i INPUT_FOLDER -o OUTPUT_FOLDER_MODEL_2 -f  0 1 2 3 4 -tr nnUNetTrainerUMambaEncNoAMP -c 3d_fullres -p nnUNetPlans --save_probabilities
```
- Ensemble
```bash
nnUNetv2_ensemble -i OUTPUT_FOLDER_MODEL_1 OUTPUT_FOLDER_MODEL_2 -o OUTPUT_FOLDER -np 8
```


