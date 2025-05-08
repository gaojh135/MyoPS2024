import os
import json
import argparse
import numpy as np
from nnUNet.nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnUNet.nnunetv2.paths import nnUNet_raw
from crop_fine import read_nifti_image, save_nifti_image

def restore_from_crop(cropped: np.ndarray, bbox: list, original_shape: list) -> np.ndarray:
    restored = np.zeros(original_shape, dtype=cropped.dtype)
    z_range, x_range, y_range = bbox
    restored[z_range[0]:z_range[1], x_range[0]:x_range[1], y_range[0]:y_range[1]] = cropped
    return restored

def convert_labels(seg: np.ndarray):
    new_seg = np.zeros_like(seg, dtype=np.int32)
    new_seg[seg == 1] = 200
    new_seg[seg == 2] = 1220
    new_seg[seg == 3] = 2221
    return new_seg

def main():
    parser = argparse.ArgumentParser(description="Restore the size of cropped prediction results and save them as CaseID_pred.nii.gz")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset ID")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory of cropped prediction results")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save restored prediction images")
    args = parser.parse_args()

    dataset = maybe_convert_to_dataset_name(args.dataset)
    dataset_dir = os.path.join(nnUNet_raw, dataset)
    bboxes_path = os.path.join(dataset_dir, "bboxes.json")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(bboxes_path, "r") as f:
        bboxes = json.load(f)

    pred_files = [f for f in os.listdir(args.input_dir) if f.endswith(".nii.gz")]

    for f in pred_files:
        case_id = f.split(".nii.gz")[0]
        if case_id not in bboxes:
            print(f"[skip] {case_id} has no bbox information")
            continue

        bbox = bboxes[case_id]["bbox"]
        orig_shape = bboxes[case_id]["original_shape"]
        cropped_pred = read_nifti_image(os.path.join(args.input_dir, f))

        cropped_pred = convert_labels(cropped_pred)
        restored = restore_from_crop(cropped_pred, bbox, orig_shape)

        if os.path.exists(os.path.join(dataset_dir, "imagesTr", f"{case_id}_0000.nii.gz")):
            ref_image_path = os.path.join(dataset_dir, "imagesTr", f"{case_id}_0000.nii.gz")
        else:
            ref_image_path = os.path.join(dataset_dir, "imagesTs", f"{case_id}_0000.nii.gz")

        # Create a separate folder for each case and save as CaseID_pred.nii.gz
        case_dir = os.path.join(args.output_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        save_path = os.path.join(case_dir, f"{case_id}_pred.nii.gz")
        save_nifti_image(restored, save_path, ref_image_path)
        print(f"Restored and saved：{save_path}")

if __name__ == "__main__":
    main()
