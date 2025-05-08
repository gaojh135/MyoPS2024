import shutil
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnUNet.nnunetv2.paths import nnUNet_raw

def convert_MyoPS_fine_segmentation(in_file: str, out_file: str) -> None:
    """
    Converts original MyoPS labels into binary fine segmentation masks (additional channel _0003 for imagesTr):
      - 200, 1220, 2221 mapped to 1
      - Others (0, 500, 600) remain 0
    """
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    for u in np.unique(img_npy):
        if u not in [0, 200, 500, 600, 1220, 2221]:
            raise RuntimeError(f"unexpected label {u} in {in_file}")
    seg = np.zeros_like(img_npy)
    seg[np.isin(img_npy, [200, 1220, 2221])] = 1
    img_out = sitk.GetImageFromArray(seg)
    img_out.CopyInformation(img)
    sitk.WriteImage(img_out, out_file)

def convert_MyoPS_fine_labels(in_file: str, out_file: str) -> None:
    """
    Converts original MyoPS labels into multiclass fine segmentation ground truth (for labelsTr):
      - 200 mapped to 1
      - 1220 mapped to 2
      - 2221 mapped to 3
      - Others (0, 500, 600) remain 0
    """
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    for u in np.unique(img_npy):
        if u not in [0, 200, 500, 600, 1220, 2221]:
            raise RuntimeError(f"unexpected label {u} in {in_file}")
    seg = np.zeros_like(img_npy)
    seg[img_npy == 200] = 1
    seg[img_npy == 1220] = 2
    seg[img_npy == 2221] = 3
    img_out = sitk.GetImageFromArray(seg)
    img_out.CopyInformation(img)
    sitk.WriteImage(img_out, out_file)


def process_test_predictions(predict_data_dir: str, output_folder: str) -> None:
    """
    Processes predictions from the test phase, converting multiclass predicted labels into binary fine segmentation masks:
      - Predicted labels 1 mapped to 1; labels 2, 3 mapped to 0
    The converted result is saved as channel _0003
    """
    nii_files = subfiles(predict_data_dir, suffix='.nii.gz', join=False)
    for f in nii_files:
        in_file = join(predict_data_dir, f)
        out_file = join(output_folder, f.replace('.nii.gz', '_0003.nii.gz'))
        img = sitk.ReadImage(in_file)
        img_npy = sitk.GetArrayFromImage(img)

        seg_new = np.zeros_like(img_npy)
        seg_new[img_npy == 1] = 1
        seg_new[np.isin(img_npy, [2, 3])] = 0

        img_corr = sitk.GetImageFromArray(seg_new)
        img_corr.CopyInformation(img)
        sitk.WriteImage(img_corr, out_file)


if __name__ == '__main__':

    data_dir = "./data/data_preprocessed"
    task_id = 401  # Dataset ID
    task_name = "myops_fine_unet_0508"
    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagesTr = join(out_base, "imagesTr")
    labelsTr = join(out_base, "labelsTr")
    imagesTs = join(out_base, "imagesTs")
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)
    maybe_mkdir_p(imagesTs)

    # Training and testing data stored under data_dir/train and data_dir/test, respectively
    case_ids_train = subdirs(join(data_dir, "train"), prefix="Case", join=False)
    case_ids_test = subdirs(join(data_dir, "test"), prefix="Case", join=False)

    print("Processing training data for fine segmentation")
    for c in case_ids_train:
        shutil.copy(join(data_dir, "train", c, c + "_C0.nii.gz"), join(imagesTr, c + '_0000.nii.gz'))
        shutil.copy(join(data_dir, "train", c, c + "_LGE.nii.gz"), join(imagesTr, c + '_0001.nii.gz'))
        shutil.copy(join(data_dir, "train", c, c + "_T2.nii.gz"), join(imagesTr, c + '_0002.nii.gz'))
        convert_MyoPS_fine_segmentation(join(data_dir, "train", c, c + "_gd.nii.gz"), join(imagesTr, c + '_0003.nii.gz'))
        convert_MyoPS_fine_labels(join(data_dir, "train", c, c + "_gd.nii.gz"), join(labelsTr, c + '.nii.gz'))

    print("Processing test data for fine segmentation")
    for c in case_ids_test:
        shutil.copy(join(data_dir, "test", c, c + "_C0.nii.gz"), join(imagesTs, c + '_0000.nii.gz'))
        shutil.copy(join(data_dir, "test", c, c + "_LGE.nii.gz"), join(imagesTs, c + '_0001.nii.gz'))
        shutil.copy(join(data_dir, "test", c, c + "_T2.nii.gz"), join(imagesTs, c + '_0002.nii.gz'))

    predict_data_dir = "./data/nnUNet_results/Dataset200_myops_coarse_90_20250421/predict/unet2d"  # Modify as needed
    process_test_predictions(predict_data_dir, imagesTs)

    generate_dataset_json(out_base,
                          channel_names={0: 'C0', 1: 'LGE', 2: 'T2', 3: 'coarse_seg'},
                          labels={
                              'background': 0,
                              'myo': (1, 2, 3),
                              'scar&edema': (2, 3),
                              'scar': (3, )
                          },
                          num_training_cases=len(case_ids_train),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3)
                          )
