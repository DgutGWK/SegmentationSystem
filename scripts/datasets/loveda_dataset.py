# SimpleLoveDADataset（只管数据）

import os
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

# 该类封装了 LoveDA 数据集的读取逻辑，支持 Urban/Rural 场景切换、离线或在线 resize、训练阶段数据增强，
# 并以 PyTorch Dataset 形式输出可直接用于 U-Net 训练与评估的数据
class SimpleLoveDADataset(Dataset):
    
    # 初始化数据集
    def __init__(
        self,
        data_root,
        mode='train',
        scene='urban',
        max_samples=None,
        target_size=256,
        augment=False,
        use_offline_resize=True,
        ignore_index=0,  # 忽略标签为0的像素（无数据区域）
        num_classes=8,    # 总共8个类别：0-7，其中0是忽略区域
        debug=False      # 调试模式
    ):
        self.mode = mode.lower()
        self.scene = scene.lower()
        self.target_size = target_size
        self.augment = augment
        self.use_offline_resize = use_offline_resize
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.debug = debug
        
        # 类别名称（根据官方文档）
        self.class_names = [
            '忽略区域',  # 0: no-data (ignore)
            '背景',      # 1: background
            '建筑',      # 2: building
            '道路',      # 3: road
            '水体',      # 4: water
            '荒地',      # 5: barren
            '森林',      # 6: forest
            '农业'       # 7: agriculture
        ]

        # 选择场景
        scenes = []
        if self.scene == 'both':
            scenes = ['Urban', 'Rural']
        elif self.scene == 'urban':
            scenes = ['Urban']
        elif self.scene == 'rural':
            scenes = ['Rural']
        else:
            raise ValueError(f"Unknown scene: {scene}")

        # 图像与 mask 文件路径收集
        self.img_files, self.mask_files = [], []
        split_dir = self.mode.capitalize()

        for s in scenes:
            # 离线 resize 目录选择
            if self.use_offline_resize:
                img_dir = os.path.join(
                    data_root, split_dir, s, f'images_{self.target_size}'
                )
                mask_dir = os.path.join(
                    data_root, split_dir, s, f'masks_{self.target_size}'
                )
            else:
                img_dir = os.path.join(
                    data_root, split_dir, s, 'images_png'
                )
                mask_dir = os.path.join(
                    data_root, split_dir, s, 'masks_png'
                )

            imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))

            # 样本筛选
            if max_samples:
                # 均匀采样
                if len(imgs) > max_samples // len(scenes):
                    step = len(imgs) // (max_samples // len(scenes))
                    imgs = imgs[::step][:max_samples // len(scenes)]
                else:
                    imgs = imgs[:max_samples // len(scenes)]

            # 图像–mask 配对
            for img in imgs:
                self.img_files.append(img)
                self.mask_files.append(
                    os.path.join(mask_dir, os.path.basename(img))
                )

        # 打印数据集信息
        if self.debug:
            print(f"[LoveDA数据集] 模式: {self.mode}, 场景: {self.scene}")
            print(f"[LoveDA数据集] 总图像数: {len(self.img_files)}")
            print(f"[LoveDA数据集] 类别数: {self.num_classes} (0-{self.num_classes-1})")
            print(f"[LoveDA数据集] 忽略索引: {self.ignore_index}")
            print(f"[LoveDA数据集] 类别含义: {self.class_names}")
            if len(self.img_files) > 0:
                print(f"[LoveDA数据集] 示例图像: {os.path.basename(self.img_files[0])}")

    def __len__(self):
        return len(self.img_files)

    # ==================== 新增：数据增强方法 ====================
    def rare_class_augmentation(self, image, mask):
        """针对稀有类别的数据增强"""
        # 定义稀有类别（根据分布统计）
        rare_classes = [4, 6, 7]  # 水体、森林、农业
        
        # 检查是否包含稀有类别
        mask_np = np.array(mask) if not isinstance(mask, np.ndarray) else mask
        
        for cls in rare_classes:
            if np.sum(mask_np == cls) > 50:  # 如果包含足够多的该类别像素
                # 随机应用颜色增强
                if random.random() > 0.5:
                    # 转换为PIL图像以便处理
                    if isinstance(image, torch.Tensor):
                        image_pil = TF.to_pil_image(image)
                    else:
                        image_pil = image
                    
                    # 随机亮度/对比度/饱和度调整
                    brightness = random.uniform(0.8, 1.2)
                    contrast = random.uniform(0.8, 1.2)
                    saturation = random.uniform(0.8, 1.2)
                    
                    image_pil = TF.adjust_brightness(image_pil, brightness)
                    image_pil = TF.adjust_contrast(image_pil, contrast)
                    image_pil = TF.adjust_saturation(image_pil, saturation)
                    
                    # 转换回原来的格式
                    if isinstance(image, torch.Tensor):
                        image = TF.to_tensor(image_pil)
                    else:
                        image = image_pil
                
                break  # 只增强第一个找到的稀有类别
        
        return image, mask
    
    def mixup_augmentation(self, image, mask, idx):
        """MixUp数据增强"""
        # 随机选择另一个样本（简化版本）
        other_idx = random.randint(0, len(self)-1)
        
        # 避免无限递归
        if other_idx == idx:
            return image, mask
        
        # 加载另一个样本
        other_image, other_mask = self._load_item(other_idx)
        
        # 随机混合系数
        lam = random.uniform(0.3, 0.7)
        
        # 混合图像
        if isinstance(image, torch.Tensor):
            mixed_image = lam * image + (1 - lam) * other_image
        else:
            # 对于PIL图像，转换为numpy处理
            image_np = np.array(image).astype(np.float32)
            other_image_np = np.array(other_image).astype(np.float32)
            mixed_image_np = lam * image_np + (1 - lam) * other_image_np
            mixed_image = Image.fromarray(mixed_image_np.astype(np.uint8))
        
        # 对于mask，使用硬标签（随机选择）
        if random.random() > 0.5:
            mixed_mask = mask
        else:
            mixed_mask = other_mask
        
        return mixed_image, mixed_mask
    
    def _load_item(self, idx):
        """内部方法：加载单个样本，避免递归"""
        # 读取图像
        image = Image.open(self.img_files[idx]).convert('RGB')
        
        # 读取mask
        mask_path = self.mask_files[idx]
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = Image.new('L', (self.target_size, self.target_size), 0)
        
        # Resize（只有非离线时）
        if not self.use_offline_resize:
            image = TF.resize(image, (self.target_size, self.target_size))
            mask = TF.resize(
                mask,
                (self.target_size, self.target_size),
                interpolation=TF.InterpolationMode.NEAREST
            )
        
        return image, mask
    # ==================== 结束：数据增强方法 ====================

    # 读取单个样本
    def __getitem__(self, idx):
        # 读取图像
        image = Image.open(self.img_files[idx]).convert('RGB')

        # 获取mask路径
        mask_path = self.mask_files[idx]

        # 读取mask
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = Image.new('L', (self.target_size, self.target_size), 0)
            if self.debug:
                print(f"[警告] Mask文件不存在: {os.path.basename(mask_path)}，使用全零mask")

        # ==================== 数据增强 ====================
        if self.mode == 'train' and self.augment:
            # 基础增强
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-30, 30)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                saturation = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)
                image = TF.adjust_saturation(image, saturation)
            
            # 新增：随机裁剪增强
            if random.random() > 0.7 and self.target_size > 200:
                crop_size = random.randint(180, self.target_size)
                i = random.randint(0, self.target_size - crop_size)
                j = random.randint(0, self.target_size - crop_size)
                
                image = TF.crop(image, i, j, crop_size, crop_size)
                mask = TF.crop(mask, i, j, crop_size, crop_size)
                
                # 重新调整到目标大小
                image = TF.resize(image, (self.target_size, self.target_size))
                mask = TF.resize(
                    mask,
                    (self.target_size, self.target_size),
                    interpolation=TF.InterpolationMode.NEAREST
                )
            
            # 新增：稀有类别增强（10%概率）
            if random.random() < 0.1:
                image, mask = self.rare_class_augmentation(image, mask)
            
            # 新增：MixUp增强（5%概率）
            if random.random() < 0.05:
                image, mask = self.mixup_augmentation(image, mask, idx)
        # ==================== 结束：数据增强 ====================

        # Resize（只有非离线时）
        if not self.use_offline_resize:
            image = TF.resize(image, (self.target_size, self.target_size))
            mask = TF.resize(
                mask,
                (self.target_size, self.target_size),
                interpolation=TF.InterpolationMode.NEAREST
            )

        # 转Tensor
        image_tensor = TF.to_tensor(image)
        
        # 处理mask：转换为numpy
        mask_np = np.array(mask, dtype=np.int64)
        
        # 验证标签范围（应该为0-7）
        min_val, max_val = mask_np.min(), mask_np.max()
        if max_val > 7 or min_val < 0:
            if self.debug and idx < 10:  # 只显示前10个警告
                print(f"[警告] {os.path.basename(mask_path)}: "
                      f"标签值超出范围: {min_val} ~ {max_val}")
            # 限制在0-7范围内
            mask_np = np.clip(mask_np, 0, 7)
        
        # 统计类别分布（调试用）
        if self.debug and idx < 5:
            unique, counts = np.unique(mask_np, return_counts=True)
            total = mask_np.size
            print(f"[样本 {idx}] {os.path.basename(self.img_files[idx])}:")
            for cls, count in zip(unique, counts):
                percentage = count / total * 100
                print(f"  类别{cls}({self.class_names[cls]}): {count}像素 ({percentage:.1f}%)")

        # 转换为Tensor
        mask_tensor = torch.from_numpy(mask_np).long()
        
        # 调试信息
        if self.debug and idx < 3:
            unique_vals = torch.unique(mask_tensor)
            unique_names = [self.class_names[val.item()] for val in unique_vals]
            print(f"[样本 {idx}] 图像: {os.path.basename(self.img_files[idx])}")
            print(f"  图像形状: {image_tensor.shape}")
            print(f"  标签形状: {mask_tensor.shape}")
            print(f"  标签范围: {mask_tensor.min().item()}-{mask_tensor.max().item()}")
            print(f"  包含类别: {sorted(unique_vals.tolist())}")
            print(f"  类别名称: {unique_names}")

        return image_tensor, mask_tensor
    
    def get_class_distribution(self, sample_count=100):
        """获取类别分布统计"""
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        
        for i in range(min(sample_count, len(self))):
            _, mask = self[i]
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
            unique, counts = np.unique(mask_np, return_counts=True)
            
            for cls, count in zip(unique, counts):
                class_counts[cls] += count
        
        total = class_counts.sum()
        if total > 0:
            distribution = class_counts / total
            print("\n类别分布统计:")
            for cls in range(self.num_classes):
                print(f"  类别{cls}({self.class_names[cls]}): {class_counts[cls]:,}像素 "
                      f"({distribution[cls]*100:.2f}%)")
        
        return class_counts

if __name__ == "__main__":
    print("=" * 70)
    print("LoveDA数据集测试")
    print("=" * 70)
    
    # 设置数据集路径
    data_root = "D:/Projects/SegmentationSystem/2021LoveDA"
    
    # 测试不同的配置
    test_configs = [
        {
            "name": "城市训练集",
            "mode": "train",
            "scene": "urban",
            "debug": True
        },
        {
            "name": "乡村训练集",
            "mode": "train",
            "scene": "rural",
            "debug": True
        },
        {
            "name": "全部验证集",
            "mode": "val",
            "scene": "both",
            "debug": True
        }
    ]
    
    for config in test_configs:
        print(f"\n测试: {config['name']}")
        print("-" * 40)
        
        try:
            # 创建数据集实例
            dataset = SimpleLoveDADataset(
                data_root=data_root,
                mode=config['mode'],
                scene=config['scene'],
                target_size=256,
                augment=False,
                use_offline_resize=False,
                debug=config['debug']
            )
            
            print(f"✓ 数据集创建成功")
            print(f"  数据集大小: {len(dataset)} 个样本")
            
            if len(dataset) > 0:
                # 测试加载前3个样本
                for i in range(min(3, len(dataset))):
                    img, mask = dataset[i]
                    print(f"\n样本 {i}:")
                    print(f"  图像形状: {img.shape}")
                    print(f"  标签形状: {mask.shape}")
                    
                    # 统计类别
                    unique_vals = torch.unique(mask)
                    print(f"  包含类别: {sorted(unique_vals.tolist())}")
                    
                    # 验证标签范围
                    if mask.min() >= 0 and mask.max() <= 7:
                        print(f"  ✓ 标签范围验证通过 (0-7)")
                    else:
                        print(f"  ✗ 标签范围异常!")
                
                # 获取类别分布
                if len(dataset) >= 50:
                    dataset.get_class_distribution(sample_count=50)
            
        except FileNotFoundError as e:
            print(f"✗ 文件未找到错误: {e}")
            print(f"  请检查数据集路径: {data_root}")
            print(f"  确保目录结构正确:")
            print(f"  {data_root}/")
            print(f"  ├── Train/")
            print(f"  │   ├── Urban/")
            print(f"  │   │   ├── images_png/")
            print(f"  │   │   └── masks_png/")
            print(f"  │   └── Rural/")
            print(f"  │       ├── images_png/")
            print(f"  │       └── masks_png/")
            print(f"  └── Val/")
            print(f"      ├── Urban/")
            print(f"      └── Rural/")
            
        except Exception as e:
            print(f"✗ 加载失败: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)