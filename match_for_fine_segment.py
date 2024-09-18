import os
import SimpleITK as sitk
import numpy as np

def compute_histogram(image, bins=256):
    image_array = sitk.GetArrayFromImage(image)
    min_val, max_val = image_array.min(), image_array.max()
    hist, _ = np.histogram(image_array, bins=bins, range=[min_val, max_val])
    hist = hist / hist.sum()  # 归一化
    return hist, min_val, max_val

def compute_average_histogram(image_files, bins=256):
    histograms = []
    min_vals, max_vals = [], []
    for image_file in image_files:
        image = sitk.ReadImage(image_file)
        hist, min_val, max_val = compute_histogram(image, bins)
        histograms.append(hist)
        min_vals.append(min_val)
        max_vals.append(max_val)
    average_histogram = np.mean(histograms, axis=0)
    avg_min_val = np.mean(min_vals)
    avg_max_val = np.mean(max_vals)
    return average_histogram, avg_min_val, avg_max_val

def save_histogram_matched_image(source_path, template_histogram, template_min, template_max):
    source_image = sitk.ReadImage(source_path)
    source_histogram, source_min, source_max = compute_histogram(source_image)

    # 计算累计分布函数（CDF）
    cdf_source = np.cumsum(source_histogram)
    cdf_template = np.cumsum(template_histogram)
    
    # 将源图像的CDF映射到模板图像的CDF
    mapping = np.interp(cdf_source, cdf_template, np.linspace(template_min, template_max, len(cdf_template)))

    source_array = sitk.GetArrayFromImage(source_image)
    # 对源图像的像素值进行映射
    matched_array = np.interp(source_array, np.linspace(source_min, source_max, len(cdf_source)), mapping)
    
    matched_image = sitk.GetImageFromArray(matched_array)
    matched_image.CopyInformation(source_image)
    sitk.WriteImage(matched_image, source_path)
    print(f"Histogram matched image saved to {source_path}")

def process_images(source_dir, template_histograms):
    for root, dirs, files in os.walk(source_dir):
        for file_name in files:
            source_path = os.path.join(root, file_name)
            
            if file_name == "crop_information.json":
                print(f"Skipping {source_path}")
            elif "labelsTr" in root:
                print(f"Skipping {source_path} (labelsTr directory)")
            elif file_name.endswith(".nii.gz"):
                if "_0003.nii.gz" in file_name:
                    print(f"Skipping {source_path}")
                else:
                    print(f"Processing {source_path}...")
                    if "_0000.nii.gz" in file_name:
                        save_histogram_matched_image(source_path, *template_histograms['_0000'])
                    elif "_0001.nii.gz" in file_name:
                        save_histogram_matched_image(source_path, *template_histograms['_0001'])
                    elif "_0002.nii.gz" in file_name:
                        save_histogram_matched_image(source_path, *template_histograms['_0002'])

if __name__ == "__main__":
    source_dir_for_template = "/home/gjh/code/U-Mamba/data/match_data/crop_all_channels_16T2_23LGE_25C0"  # 替换为模板源目录
    source_dir_for_matching = "/home/gjh/code/U-Mamba/data/nnUNet_raw/Dataset144_MyoPS2024_fine_seg_crop_match_no0255_label_umamba_stage4_epoch150_0914"  # 替换为待匹配源目录

    template_files_0000 = []
    template_files_0001 = []
    template_files_0002 = []
    
    for root, dirs, files in os.walk(source_dir_for_template):
        for file_name in files:
            if file_name.endswith("_0000.nii.gz"):
                template_files_0000.append(os.path.join(root, file_name))
            elif file_name.endswith("_0001.nii.gz"):
                template_files_0001.append(os.path.join(root, file_name))
            elif file_name.endswith("_0002.nii.gz"):
                template_files_0002.append(os.path.join(root, file_name))
    
    if not template_files_0000:
        raise ValueError("模板源目录中没有找到 _0000.nii.gz 文件。")
    if not template_files_0001:
        raise ValueError("模板源目录中没有找到 _0001.nii.gz 文件。")
    if not template_files_0002:
        raise ValueError("模板源目录中没有找到 _0002.nii.gz 文件。")

    template_histogram_0000 = compute_average_histogram(template_files_0000)
    template_histogram_0001 = compute_average_histogram(template_files_0001)
    template_histogram_0002 = compute_average_histogram(template_files_0002)
    
    template_histograms = {
        '_0000': template_histogram_0000,
        '_0001': template_histogram_0001,
        '_0002': template_histogram_0002
    }
    
    process_images(source_dir_for_matching, template_histograms)
