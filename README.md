# Code for Code for CARE2024-MyoPS

## Installation

Requirements: `Ubuntu 22.04`, `CUDA 12.1`

```bash
conda create -n rgumamba python=3.11 -y
conda activate rgumamba
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install causal-conv1d
pip install mamba-ssm
git clone https://github.com/gaojh135/MyoPS2024.git
cd MyoPS2024/nnUNet
pip install -e .
```
## Path settings

```python
base = './data'
nnUNet_raw = join(base, 'nnUNet_raw') # or change to os.environ.get('nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # or change to os.environ.get('nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results') # or change to os.environ.get('nnUNet_results')
```

## Data Structure
```
.
├── data
│   ├── data_raw
│   ├── CARE2024_MyoPS++
│   │   ├── aligned_train
……………………
│   ├── CARE2024_MyoPS++_valid
│   │   ├── aligned_train
……………………
```

## Pipeline

### 1. Coarse-seg

```bash
python copy_data.py
python data_converison_coarse.py # You can change task_id & task_name if you want
python crop_coarse.py -d DATASET_ID
```

#### Pre-process & Training

```bash
# Pre‑process
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Five‑fold cross‑validation
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train DATASET_ID 2d         $FOLD -tr UMambaEncTrainer --npz
  nnUNetv2_train DATASET_ID 3d_fullres $FOLD -tr UMambaEncTrainer --npz
done
```

#### Find the best configuration & Run inference
```bash
nnUNetv2_find_best_configuration DATASET_ID -c 2d 3d_fullres
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f 0 1 2 3 4 -tr UMambaEncTrainer
```

#### Ensemble & Post‑process
```bash
nnUNetv2_ensemble -i FOLDER1 FOLDER2 -o OUTPUT_FOLDER -np NUM_PROCESSES
nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
```

```bash
# Restore coarse predictions to original spacing
python restore_coarse.py -d Dataset_ID -i input_folder -o output_folder
```

## 2. Fine-seg 

```bash
python data_converison_fine_region.py # You can change task_id & task_name if you want
python crop_fine.py -d DATASET_ID
```

### Model Training
Follow the same pre-processing and training procedures as described in the Coarse-seg section above, including preprocess, training, configuration selection, and inference:

```bash
# Restore coarse predictions to original spacing
python restore_fine.py -d Dataset_ID -i input_folder -o output_folder
```
