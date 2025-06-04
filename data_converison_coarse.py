import numpy as np
import shutil
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnUNet.nnunetv2.paths import nnUNet_raw

def copy_MyoPS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. MyoPS_data is 0, 200, 500, 600, 1220, 2221 -> we make that into 0, 1, 2, 3, 1, 1
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 200, 500, 600, 1220, 2221]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 200] = 1
    seg_new[img_npy == 500] = 2
    seg_new[img_npy == 600] = 3
    seg_new[img_npy == 1220] = 1
    seg_new[img_npy == 2221] = 1
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

if __name__ == '__main__':
    data_dir = "./data/data_preprocessed/"

    task_id = 400 # Dataset ID
    task_name = "myops_coarse_80_20250508"
    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base, "imagesTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)

    case_ids_train = subdirs(join(data_dir, "train"), prefix='Case', join=False)
    case_ids_test = subdirs(join(data_dir, "test"), prefix='Case', join=False)

    print("copying train data")
    for c in case_ids_train:
        shutil.copy(join(data_dir, "train", c, c + "_C0.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(join(data_dir, "train", c, c + "_LGE.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
        shutil.copy(join(data_dir, "train", c, c + "_T2.nii.gz"), join(imagestr, c + '_0002.nii.gz'))

        copy_MyoPS_segmentation_and_convert_labels_to_nnUNet(join(data_dir, "train", c, c + "_gd.nii.gz"),
                                                             join(labelstr, c + '.nii.gz'))
    
    print("copying test data")
    for c in case_ids_test:
        shutil.copy(join(data_dir, "test", c, c + "_C0.nii.gz"), join(imagests, c + '_0000.nii.gz'))
        shutil.copy(join(data_dir, "test", c, c + "_LGE.nii.gz"), join(imagests, c + '_0001.nii.gz'))
        shutil.copy(join(data_dir, "test", c, c + "_T2.nii.gz"), join(imagests, c + '_0002.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'C0', 1: 'LGE', 2: 'T2'},
                          labels={
                              'background': 0,
                              'myo': 1,
                              'lv': 2,
                              'rv': 3
                          },
                          num_training_cases=len(case_ids_train),
                          file_ending='.nii.gz'
                          )
