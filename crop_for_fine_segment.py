import os
from collections import OrderedDict
import SimpleITK as sitk
from pymic.io.image_read_write import save_array_as_nifty_volume
from pymic.util.image_process import crop_ND_volume_with_bounding_box, get_ND_bounding_box
from batchgenerators.utilities.file_and_folder_operations import save_json

def crop_images(images_dir, images_output_dir, masks_dir, mask_suffix=None, masks_output_dir=None, margin=[0, 30, 30]):
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    if masks_output_dir is not None and (not os.path.exists(masks_output_dir)):
        os.makedirs(masks_output_dir)
    
    masks_list = os.listdir(masks_dir)
    masks_list.sort()
    json_dict = OrderedDict()
    
    for mask in masks_list:
        mask_path = os.path.join(masks_dir, mask)
        if mask.endswith('.nii.gz'):
            mask_sitk = sitk.ReadImage(mask_path)
            mask_npy = sitk.GetArrayFromImage(mask_sitk)
            mask_shape = mask_npy.shape
            
            # 获取裁剪边界框
            crop_bbox_min, crop_bbox_max = get_ND_bounding_box(mask_npy, margin=margin)
            crop_bbox_min[0] = 0  # 不在第一个维度上裁剪
            crop_bbox_max[0] = mask_shape[0]
            
            # 保存裁剪信息
            json_dict[mask_path] = {"crop_bbox_min": crop_bbox_min, "crop_bbox_max": crop_bbox_max}
            
            # 裁剪对应的图像并保存
        if mask_suffix:  # 如果有指定后缀（用于 imagesTr 和 imagesTs）
            for suffix in mask_suffix:
                image = mask.replace('.nii.gz', f'{suffix}.nii.gz')
                image_path = os.path.join(images_dir, image)
                print(f"Cropping {image_path}")  # Log the cropping operation
                if not os.path.exists(image_path):
                    print(f"File {image_path} does not exist")  # Log missing file
                    continue
                image_sitk = sitk.ReadImage(image_path)
                image_npy = sitk.GetArrayFromImage(image_sitk)
                image_output_npy = crop_ND_volume_with_bounding_box(image_npy, crop_bbox_min, crop_bbox_max)
                save_array_as_nifty_volume(image_output_npy, os.path.join(images_output_dir, image), image_path)
        else:  # 对于没有后缀的文件（用于 labelsTr）
                image_output_npy = crop_ND_volume_with_bounding_box(mask_npy, crop_bbox_min, crop_bbox_max)
                save_array_as_nifty_volume(image_output_npy, os.path.join(images_output_dir, mask), mask_path)

    # 保存裁剪信息到JSON文件
    save_json(json_dict, os.path.join(images_output_dir, "crop_information.json"))
    if masks_output_dir is not None:
        save_json(json_dict, os.path.join(masks_output_dir, "crop_information.json"))

def main():
    predict_ensemble_dir = '/home/gjh/code/U-Mamba/data/nnUNet_results/Dataset141_MyoPS2024_coarse_seg_match_no0255_crop_label_umamba_stage5_epoch150_0911/predict_ensemble_PP'
    nnunet_raw_dir = '/home/gjh/code/U-Mamba/data/nnUNet_raw/Dataset144_MyoPS2024_fine_seg_crop_match_no0255_label_umamba_stage4_epoch150_0914'
    #nnunet_outpath_dir = '/home/gjh/code/U-Mamba/test/Dataset013_MyoPS2024_fine_seg'(若无需覆盖，下方面路径修改即可，imagesTr_dir、imagesTs_dir、labelsTr_dir)

    #imagesTr_outpath_dir = os.path.join(nnunet_outpath_dir, 'imagesTr')
    #imagesTs_outpath_dir = os.path.join(nnunet_outpath_dir, 'imagesTs')
    #labelsTr_outpath_dir = os.path.join(nnunet_outpath_dir, 'labelsTr')

    imagesTr_dir = os.path.join(nnunet_raw_dir, 'imagesTr')
    imagesTs_dir = os.path.join(nnunet_raw_dir, 'imagesTs')
    labelsTr_dir = os.path.join(nnunet_raw_dir, 'labelsTr')
    
    # 处理训练集图像的裁剪
    crop_images(imagesTr_dir, imagesTr_dir, labelsTr_dir, mask_suffix=['_0000', '_0001', '_0002', '_0003'], masks_output_dir=None)

    # 处理测试集图像的裁剪
    crop_images(imagesTs_dir, imagesTs_dir, predict_ensemble_dir, mask_suffix=['_0000', '_0001', '_0002', '_0003'], masks_output_dir=None)
    
    # 裁剪训练集标签
    crop_images(labelsTr_dir, labelsTr_dir, labelsTr_dir, mask_suffix=None, masks_output_dir=labelsTr_dir)

if __name__ == "__main__":
    main()
