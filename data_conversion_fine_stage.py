import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def copy_MyoPS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. MyoPS is 0, 200, 500, 600, 1220, 2221 -> we make that into 0, 1, 0, 0, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 200, 500, 600, 1220, 2221]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 200] = 1
    seg_new[img_npy == 500] = 0
    seg_new[img_npy == 600] = 0
    seg_new[img_npy == 1220] = 2
    seg_new[img_npy == 2221] = 3

    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def convert_labels_back_to_MyoPS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 200
    new_seg[seg == 0] = 500
    new_seg[seg == 0] = 600
    new_seg[seg == 2] = 1221
    new_seg[seg == 3] = 2221
    return new_seg


def load_convert_labels_back_to_MyoPS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_MyoPS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_MyoPS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_MyoPS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


def convert_and_save_0003(in_file: str, out_file: str) -> None:
    # Function to convert labels in 0003 channel and save
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    
    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 200] = 1
    seg_new[img_npy == 500] = 0
    seg_new[img_npy == 600] = 0
    seg_new[img_npy == 1220] = 1  # Based on provided mapping rules
    seg_new[img_npy == 2221] = 1  # Based on provided mapping rules

    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


if __name__ == '__main__':
    MyoPS_data_dir = '/home/gjh/code/U-Mamba/data/train_data_pp'
    test_data_dir = '/home/gjh/code/U-Mamba/data/test_data_pp'
    predict_data_dir = '/home/gjh/code/U-Mamba/data/nnUNet_results/Dataset151_MyoPS2024_coarse_seg_match_no0255_crop_label_nnunet_stage5_epoch150_0912/predict_ensemble_PP'

    task_id = 152
    task_name = "MyoPS2024_fine_seg_crop_match_region_nnunet_stage4_epoch150_0912"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base, "imagesTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)

    # Process training data
    case_ids = subdirs(MyoPS_data_dir, prefix='Case', join=False)

    for c in case_ids:
        shutil.copy(join(MyoPS_data_dir, c, c + "_C0.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(MyoPS_data_dir, c, c + "_LGE.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
        shutil.copy(join(MyoPS_data_dir, c, c + "_T2.nii.gz"), join(imagestr, c + '_0002.nii.gz'))

        copy_MyoPS_segmentation_and_convert_labels_to_nnUNet(join(MyoPS_data_dir, c, c + "_gd.nii.gz"),
                                                             join(labelstr, c + '.nii.gz'))

        # 保存并转换原始标签为imagesTr的0003
        convert_and_save_0003(join(MyoPS_data_dir, c, c + "_gd.nii.gz"), join(imagestr, c + '_0003.nii.gz'))
    
    # Process test data
    test_case_ids = subdirs(test_data_dir, prefix='Case', join=False)

    for c in test_case_ids:
        shutil.copy(join(test_data_dir, c, c + "_C0.nii.gz"), join(imagests, c + '_0000.nii.gz'))
        shutil.copy(join(test_data_dir, c, c + "_LGE.nii.gz"), join(imagests, c + '_0001.nii.gz'))
        shutil.copy(join(test_data_dir, c, c + "_T2.nii.gz"), join(imagests, c + '_0002.nii.gz'))
    
    # 将预测结果转换并保存到imagesTs的0003
    nii_files = subfiles(predict_data_dir, suffix='.nii.gz', join=False)
    for f in nii_files:
        in_file = join(predict_data_dir, f)
        out_file = join(imagests, f.replace('.nii.gz', '_0003.nii.gz'))
        img = sitk.ReadImage(in_file)
        img_npy = sitk.GetArrayFromImage(img)
        
        seg_new = np.zeros_like(img_npy)
        seg_new[img_npy == 1] = 1
        seg_new[img_npy == 2] = 0
        seg_new[img_npy == 3] = 0
        seg_new[img_npy == 4] = 1  # Adjust according to conversion rules
        seg_new[img_npy == 5] = 1  # Adjust according to conversion rules
        
        img_corr = sitk.GetImageFromArray(seg_new)
        img_corr.CopyInformation(img)
        sitk.WriteImage(img_corr, out_file)

    generate_dataset_json(out_base,
                          channel_names={0: 'C0', 1: 'LGE', 2: 'T2', 3: 'coarse_seg'},
                          labels={
                              'background': 0,
                              'myo': (1, 2, 3),
                              'scar&edema': (2, 3),
                              'scar': (3, )
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3)
                          )
