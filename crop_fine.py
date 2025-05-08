import os
import argparse
import json
import SimpleITK as sitk
from tqdm import tqdm
from nnUNet.nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnUNet.nnunetv2.paths import nnUNet_raw
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, pad_bbox

def read_nifti_image(file_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))

def save_nifti_image(image, output_path, reference_path):
    reference = sitk.ReadImage(reference_path)
    img = sitk.GetImageFromArray(image)
    img.SetSpacing(reference.GetSpacing())
    img.SetOrigin(reference.GetOrigin())
    img.SetDirection(reference.GetDirection())
    sitk.WriteImage(img, output_path)

def crop_image_and_label_based_on_mask(image_paths, label_path, output_image_paths, output_label_path, extend=30):
    label = read_nifti_image(label_path)
    bbox = get_bbox_from_mask(label)
    extended_bbox = pad_bbox(bbox, pad_amount=[0, extend, extend], array_shape=label.shape)

    for image_path, output_image_path in zip(image_paths, output_image_paths):
        image = read_nifti_image(image_path)
        cropped_image = crop_to_bbox(image, extended_bbox)
        save_nifti_image(cropped_image, output_image_path, image_path)

    cropped_label = crop_to_bbox(label, extended_bbox)
    save_nifti_image(cropped_label, output_label_path, label_path)

    return extended_bbox, label.shape

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    args = parser.parse_args()

    dataset = maybe_convert_to_dataset_name(args.dataset)
    dataset_dir = os.path.join(nnUNet_raw, dataset)
    imagesTr_dir = os.path.join(dataset_dir, 'imagesTr')
    imagesTs_dir = os.path.join(dataset_dir, 'imagesTs')
    labelsTr_dir = os.path.join(dataset_dir, 'labelsTr')

    bboxes_dict = {}

    if os.path.isdir(imagesTr_dir) and os.path.isdir(labelsTr_dir):
        for f in tqdm(os.listdir(imagesTr_dir), desc="crop train data"):
            if not f.endswith('_0000.nii.gz'):
                continue
            case_id = f.split('_')[0]
            image_paths = [os.path.join(imagesTr_dir, f"{case_id}_{i:04d}.nii.gz") for i in range(4)]
            label_path = os.path.join(labelsTr_dir, f"{case_id}.nii.gz")
            if not os.path.exists(label_path):
                continue
            bbox, orig_shape = crop_image_and_label_based_on_mask(image_paths, label_path, image_paths, label_path)
            bboxes_dict[case_id] = {"bbox": bbox, "original_shape": orig_shape}

    if os.path.isdir(imagesTs_dir):
        for f in tqdm(os.listdir(imagesTs_dir), desc="crop test data"):
            if not f.endswith('_0000.nii.gz'):
                continue
            case_id = f.split('_')[0]
            image_paths = [os.path.join(imagesTs_dir, f"{case_id}_{i:04d}.nii.gz") for i in range(4)]
            label_path = os.path.join(imagesTs_dir, f"{case_id}_0003.nii.gz")
            if not os.path.exists(label_path):
                continue
            bbox, orig_shape = crop_image_and_label_based_on_mask(image_paths, label_path, image_paths, label_path)
            bboxes_dict[case_id] = {"bbox": bbox, "original_shape": orig_shape}

    with open(os.path.join(dataset_dir, "bboxes.json"), 'w') as f:
        json.dump(bboxes_dict, f, indent=2)

if __name__ == "__main__":
    main()
