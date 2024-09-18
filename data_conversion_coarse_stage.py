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
    # nnUNet wants the labels to be continuous. MyoPS is 0, 200, 500, 600, 1220, 2221 -> we make that into 0, 1, 2, 3, 1, 1
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 200, 500, 600, 1220, 2221]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 200] = 1
    seg_new[img_npy == 1220] = 1
    seg_new[img_npy == 2221] = 1
    seg_new[img_npy == 500] = 2
    seg_new[img_npy == 600] = 3
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def convert_labels_back_to_MyoPS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 200
    new_seg[seg == 1] = 1221
    new_seg[seg == 1] = 2221
    new_seg[seg == 2] = 500
    new_seg[seg == 3] = 600
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


if __name__ == '__main__':
    MyoPS_data_dir = '/home/gjh/code/U-Mamba/data/train_data_pp'
    test_data_dir = '/home/gjh/code/U-Mamba/data/test_data_pp'

    task_id = 141
    task_name = "MyoPS2024_coarse_seg_match_no0255_crop_label_umamba_stage5_epoch150_0911"

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
    
    # Process test data
    test_case_ids = subdirs(test_data_dir, prefix='Case', join=False)

    for c in test_case_ids:
        shutil.copy(join(test_data_dir, c, c + "_C0.nii.gz"), join(imagests, c + '_0000.nii.gz'))
        shutil.copy(join(test_data_dir, c, c + "_LGE.nii.gz"), join(imagests, c + '_0001.nii.gz'))
        shutil.copy(join(test_data_dir, c, c + "_T2.nii.gz"), join(imagests, c + '_0002.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'C0', 1: 'LGE', 2: 'T2'},
                          labels={
                              'background': 0,
                              'myops': 1,
                              'lv': 2,
                              'rv': 3
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          )
