# 平衡采样器

import torch
import random
import numpy as np
from torch.utils.data import Sampler

# 平衡批次采样器
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_classes=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # 为每个类别收集样本索引（排除类别0和1）
        self.class_indices = {i: [] for i in range(2, num_classes)}
        
        print("构建平衡采样器...")
        for idx in range(len(dataset)):
            _, mask = dataset[idx]
            unique_classes = torch.unique(mask)
            
            for cls in unique_classes:
                cls_int = cls.item()
                if cls_int >= 2:  # 只关注非背景、非忽略类别
                    self.class_indices[cls_int].append(idx)
        
        print("采样器构建完成。各类别样本数:")
        for cls in sorted(self.class_indices.keys()):
            print(f"  类别{cls}: {len(self.class_indices[cls])} 个样本")
    
    def __iter__(self):
        n_batches = len(self.dataset) // self.batch_size
        
        for _ in range(n_batches):
            batch = []
            
            # 确保包含稀有类别
            rare_classes = [7, 4, 6]  # 农业、水体、森林
            for cls in rare_classes:
                if self.class_indices[cls] and len(batch) < self.batch_size // 3:
                    batch.append(random.choice(self.class_indices[cls]))
            
            # 添加常见类别
            common_classes = [2, 3, 5]  # 建筑、道路、荒地
            for cls in common_classes:
                if self.class_indices[cls] and len(batch) < self.batch_size * 2 // 3:
                    batch.append(random.choice(self.class_indices[cls]))
            
            # 用随机样本填充
            while len(batch) < self.batch_size:
                batch.append(random.randint(0, len(self.dataset)-1))
            
            random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size