import os
import json
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm
import subprocess

# ================= 1. 设置环境变量 =================
os.environ['nnUNet_raw'] = '/root/autodl-tmp/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/root/autodl-tmp/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/root/autodl-tmp/nnUNet_results'

# ================= 2. 格式转换 =================
RAW_DATA_DIR = "/root/autodl-tmp/UCSD_RAW_DATA"
NNUNET_RAW_DIR = "/root/autodl-tmp/nnUNet_raw/Dataset001_UCSD_BMETS"
LABEL_SUFFIX = "-seg.nii.gz"

MODALITY_MAP = {
    "_0000": "-t1n.nii.gz",
    "_0001": "-t1c.nii.gz",
    "_0002": "-t2w.nii.gz",
    "_0003": "-t2f.nii.gz"
}

imagesTr_dir = os.path.join(NNUNET_RAW_DIR, "imagesTr")
labelsTr_dir = os.path.join(NNUNET_RAW_DIR, "labelsTr")
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

# 终极防御版全零图像生成器
def generate_zero_image(reference_img_path, output_path):
    if reference_img_path and os.path.exists(reference_img_path):
        try:
            ref_img = nib.load(reference_img_path)
            shape, affine = ref_img.shape, ref_img.affine
        except:
            shape, affine = (128, 128, 128), np.eye(4)
    else:
        shape, affine = (128, 128, 128), np.eye(4)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), output_path)

# 过滤掉隐藏文件夹 (比如 .git)
case_folders = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d)) and not d.startswith('.')]
num_samples = 0

print(">>> 开始格式转换...")
for case_id in tqdm(case_folders, desc="转换数据"):
    case_path = os.path.join(RAW_DATA_DIR, case_id)
    label_path = os.path.join(case_path, f"{case_id}{LABEL_SUFFIX}")
    
    if not os.path.exists(label_path):
        continue

    reference_path = None
    for mod_suffix in MODALITY_MAP.values():
        test_path = os.path.join(case_path, f"{case_id}{mod_suffix}")
        # 检查文件不仅存在，而且大小大于 0
        if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
            reference_path = test_path
            break

    for channel_idx, mod_suffix in MODALITY_MAP.items():
        src_img = os.path.join(case_path, f"{case_id}{mod_suffix}")
        dst_img = os.path.join(imagesTr_dir, f"{case_id}{channel_idx}.nii.gz")
        
        # 如果原图存在且大于 0 字节，则复制，否则填零
        if os.path.exists(src_img) and os.path.getsize(src_img) > 0:
            shutil.copy(src_img, dst_img)
        else:
            generate_zero_image(reference_path, dst_img)

    dst_label = os.path.join(labelsTr_dir, f"{case_id}.nii.gz")
    shutil.copy(label_path, dst_label)
    num_samples += 1

dataset_json = {
    "channel_names": {"0": "T1", "1": "T1c", "2": "T2", "3": "FLAIR"},
    "labels": {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4},
    "numTraining": num_samples,
    "file_ending": ".nii.gz"
}
with open(os.path.join(NNUNET_RAW_DIR, 'dataset.json'), 'w') as f:
    json.dump(dataset_json, f, indent=4)

print(f"✅ 格式转换完成！共生成 {num_samples} 个样本。")

# ================= 3. 运行预处理 =================
print("\n>>> 开始数据预处理...")
cmd = ["nnUNetv2_plan_and_preprocess", "-d", "1", "-np", "8", "--verify_dataset_integrity"]
subprocess.run(cmd, check=True)

# ================= 4. 启动 3D 训练 =================
print("\n>>> 🚀 启动 3D 模型训练 (Fold 0)...")
cmd = ["nnUNetv2_train", "1", "3d_fullres", "0", "--npz"]
subprocess.run(cmd, check=True)

print("🎉 训练结束！")