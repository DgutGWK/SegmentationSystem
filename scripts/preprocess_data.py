# 数据预处理脚本
# 对 LoveDA 数据集进行离线图像尺寸统一化处理

import os
from PIL import Image
from tqdm import tqdm

TARGET_SIZE = 256

# 将某个文件夹下的所有 PNG 文件缩放到指定尺寸，并保存到新的目标文件夹
def resize_folder(src_dir, dst_dir, is_mask=False):
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith('.png')]

    for f in tqdm(files, desc=f"Resizing {os.path.basename(dst_dir)}"):
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)

        img = Image.open(src_path)
        if is_mask:
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
        else:
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)

        img.save(dst_path)

# 针对 LoveDA 数据集的每个场景（Urban / Rural）和每个 split（train / val），分别处理图像和 mask
def process_scene(root, split, scene):
    img_src = os.path.join(root, split, scene, "images_png")
    mask_src = os.path.join(root, split, scene, "masks_png")

    img_dst = os.path.join(root, split, scene, f"images_{TARGET_SIZE}")
    mask_dst = os.path.join(root, split, scene, f"masks_{TARGET_SIZE}")

    resize_folder(img_src, img_dst, is_mask=False)
    resize_folder(mask_src, mask_dst, is_mask=True)

if __name__ == "__main__":
    data_root = "D:/Projects/SegmentationSystem/2021LoveDA"

    for split in ["train", "val"]:
        for scene in ["Urban", "Rural"]:
            print(f"\nProcessing {split}/{scene}")
            process_scene(data_root, split, scene)

    print("\n✅ 离线 Resize 完成")
