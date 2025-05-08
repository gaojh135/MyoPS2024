import os
import argparse
import json
import SimpleITK as sitk
from tqdm import tqdm
from nnUNet.nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnUNet.nnunetv2.paths import nnUNet_raw
from acvl_utils.cropping_and_padding.bounding_boxes import crop_to_bbox

def read_nifti_image(path):
    return sitk.ReadImage(path)

def save_nifti_image(array, reference, output_path):
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(reference.GetSpacing())
    img.SetOrigin(reference.GetOrigin())
    img.SetDirection(reference.GetDirection())
    sitk.WriteImage(img, output_path)

def crop_xy_inner(image: sitk.Image, crop_margin: int = 40):
    array = sitk.GetArrayFromImage(image)  # (Z, X, Y)
    z, x, y = array.shape
    bbox = [[0, z], [crop_margin, x - crop_margin], [crop_margin, y - crop_margin]]
    cropped = crop_to_bbox(array, bbox)
    return cropped, bbox, list(array.shape)

def process_folder(folder_path, bboxes_dict, crop_margin = 40):
    if not os.path.isdir(folder_path):
        return
    for fname in tqdm(os.listdir(folder_path), desc=f"Cropping {os.path.basename(folder_path)}"):
        if not fname.endswith(".nii.gz"):
            continue
        case_id = fname.split(".nii.gz")[0].split("_")[0]
        file_path = os.path.join(folder_path, fname)
        image = read_nifti_image(file_path)
        cropped_array, bbox, original_shape = crop_xy_inner(image, crop_margin)
        save_nifti_image(cropped_array, image, file_path)
        bboxes_dict[case_id] = {"bbox": bbox, "original_shape": original_shape}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    args = parser.parse_args()

    dataset = maybe_convert_to_dataset_name(args.dataset)
    dataset_dir = os.path.join(nnUNet_raw, dataset)
    bboxes_dict = {}

    for subdir in ["imagesTr", "imagesTs", "labelsTr"]:
        folder = os.path.join(dataset_dir, subdir)
        process_folder(folder, bboxes_dict, crop_margin=15)

    with open(os.path.join(dataset_dir, "bboxes.json"), "w") as f:
        json.dump(bboxes_dict, f, indent=2)

if __name__ == "__main__":
    main()
