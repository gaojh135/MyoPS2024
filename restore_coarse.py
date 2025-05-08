import os
import json
import argparse
import numpy as np
import SimpleITK as sitk
from nnUNet.nnunetv2.paths import nnUNet_raw
from nnUNet.nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

def read_nifti_image(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def save_nifti_image(array, output_path, reference_path):
    ref = sitk.ReadImage(reference_path)
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(ref.GetSpacing())
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())
    sitk.WriteImage(img, output_path)

def restore_from_crop(cropped: np.ndarray, bbox: list, original_shape: list) -> np.ndarray:
    restored = np.zeros(original_shape, dtype=cropped.dtype)
    z, x, y = bbox
    restored[z[0]:z[1], x[0]:x[1], y[0]:y[1]] = cropped
    return restored

def main():
    parser = argparse.ArgumentParser(description="Restore cropped prediction to original shape")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    args = parser.parse_args()

    dataset = maybe_convert_to_dataset_name(args.dataset)
    dataset_dir = os.path.join(nnUNet_raw, dataset)
    bboxes_path = os.path.join(dataset_dir, "bboxes.json")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(bboxes_path, "r") as f:
        bboxes = json.load(f)

    for fname in os.listdir(args.input_dir):
        if not fname.endswith(".nii.gz"):
            continue
        case_id = fname.split(".nii.gz")[0]
        if case_id not in bboxes:
            print(f"[skip] {case_id} has no bbox info")
            continue

        bbox = bboxes[case_id]["bbox"]
        orig_shape = bboxes[case_id]["original_shape"]
        cropped = read_nifti_image(os.path.join(args.input_dir, fname))
        restored = restore_from_crop(cropped, bbox, orig_shape)

        # 参考 imagesTr 或 imagesTs
        if os.path.exists(os.path.join(dataset_dir, "imagesTr", f"{case_id}_0000.nii.gz")):
            ref_path = os.path.join(dataset_dir, "imagesTr", f"{case_id}_0000.nii.gz")
        else:
            ref_path = os.path.join(dataset_dir, "imagesTs", f"{case_id}_0000.nii.gz")

        out_path = os.path.join(args.output_dir, f"{case_id}_restored.nii.gz")
        save_nifti_image(restored, out_path, ref_path)
        print(f"[OK] Restored: {out_path}")

if __name__ == "__main__":
    main()
