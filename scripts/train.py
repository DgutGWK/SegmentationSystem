# 训练 U-Net 模型
# 在 LoveDA 训练集上训练一个 U-Net 语义分割模型，并在每个 epoch 后保存最新模型参数

import os
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from datasets.loveda_dataset import SimpleLoveDADataset
from models.unet import UNet

# 计算 IoU
def compute_iou(pred, target, num_classes=8):
    ious = []
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    for cls in range(1, num_classes):  # 从1开始，跳过忽略区域
        pred_cls = pred_flat == cls
        target_cls = target_flat == cls
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        
        if union > 0:
            ious.append(intersection / union)
    
    return np.mean(ious) if ious else 0.0

# 获取项目根目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 新增：改进的损失函数 ====================
# 引入 Focal Loss 和边界损失以提升模型对难分类样本和边界区域的关注
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()

# 边界损失，关注类别边界
def boundary_loss(pred, target):

    # 计算边界权重图
    kernel = torch.ones(1, 1, 3, 3).to(pred.device)
    target_expanded = target.unsqueeze(1).float()
    
    # 使用卷积检测边界
    smoothed = F.conv2d(target_expanded, kernel, padding=1)
    boundary = (smoothed > 0) & (smoothed < 9)  # 边界像素
    
    # 在边界处增加损失权重
    boundary_weight = boundary.float() * 3.0 + 1.0
    return (F.cross_entropy(pred, target, reduction='none') * boundary_weight.squeeze()).mean()

# 组合损失函数
def compute_combined_loss(outputs, masks, class_weights, use_focal=True, use_boundary=True):

    # 基础交叉熵损失
    ce_loss = F.cross_entropy(outputs, masks, weight=class_weights)
    
    total_loss = ce_loss
    
    # 可选：添加Focal Loss
    if use_focal:
        focal = focal_loss(outputs, masks, alpha=0.25, gamma=2.0)
        total_loss = total_loss + 0.3 * focal
    
    # 可选：添加边界损失
    if use_boundary:
        boundary = boundary_loss(outputs, masks)
        total_loss = total_loss + 0.1 * boundary
    
    return total_loss

# Dice Loss
def dice_loss(pred, target, smooth=1e-6):

    pred = torch.softmax(pred, dim=1)
    num_classes = pred.shape[1]

    total_loss = 0
    for cls in range(num_classes):
        pred_cls = pred[:, cls]
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        total_loss += (1 - dice)
    
    return total_loss / num_classes

# 基于真实标签分布计算类别权重
def compute_class_weights_from_distribution():
    distribution = {
        0: 0.0347,   # 忽略区域 - 权重为0
        1: 0.4821,   # 背景 - 低权重
        2: 0.2346,   # 建筑 - 中等权重
        3: 0.0825,   # 道路 - 较高权重
        4: 0.0378,   # 水体 - 高权重
        5: 0.0656,   # 荒地 - 较高权重
        6: 0.0491,   # 森林 - 高权重
        7: 0.0136    # 农业 - 最高权重
    }
    
    weights = torch.zeros(8)
    for cls, freq in distribution.items():
        if cls == 0:  # 忽略区域
            weights[cls] = 0.0
        elif freq > 0:
            # 使用逆频率加权，更关注稀有类别
            weights[cls] = 1.0 / (freq + 0.01)
        else:
            weights[cls] = 1.0
    
    # 归一化（排除类别0）
    weights[1:] = weights[1:] / weights[1:].sum() * 7
    
    return weights
# ==================== 结束：改进的损失函数 ====================

def main():
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # ==================== 优化后的训练配置 ====================
    config = dict(
        data_root='D:/Projects/SegmentationSystem/2021LoveDA',
        scene='both',
        target_size=256,
        train_samples=300,
        val_samples=60,
        batch_size=6,
        epochs=25,
        lr=2e-4,
        weight_decay=1e-4,
        early_stop_patience=10,
        
        # 新增：类别平衡策略
        use_balanced_sampling=False,    # 平衡采样
        use_class_weights=True,        # 类别权重
        use_focal_loss=True,           # Focal Loss
        
        # 数据增强配置
        augment_config={
            'flip_prob': 0.3,
            'rotate_prob': 0.2,
            'color_jitter_prob': 0.1,
            'crop_prob': 0,
            'rare_class_aug_prob': 0,  # 稀有类别增强概率
        },
        
        # 损失函数权重
        loss_weights={
            'ce': 1.0,      # 交叉熵损失权重
            'focal': 0.2,   # Focal Loss权重
            'boundary': 0.05, # 边界损失权重
            'dice': 0.2     # Dice Loss权重
        },

        # CPU优化设置
        num_workers=0,          # Windows上设为0避免问题
        pin_memory=False,
        
        # 内存优化
        use_gradient_checkpointing=False,  # CPU上不需要
        use_mixed_precision=False,         # CPU上不支持混合精度
    )
    # ==================== 结束：训练配置 ====================

    # 加载训练和验证数据集
    print("准备数据...")
    train_dataset = SimpleLoveDADataset(
        data_root=config['data_root'],
        mode='train',
        scene=config['scene'],
        target_size=config['target_size'],
        max_samples=config['train_samples'],
        augment=True,
        use_offline_resize=True
    )

    val_dataset = SimpleLoveDADataset(
        data_root=config['data_root'],
        mode='val',
        scene=config['scene'],
        target_size=config['target_size'],
        max_samples=config['val_samples'],
        augment=False,
        use_offline_resize=True
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # ==================== 数据加载器（支持平衡采样） ====================
    if config.get('use_balanced_sampling', False):
        print("使用平衡采样策略...")
        # 注意：需要先创建balanced_sampler.py文件
        try:
            from samplers.balanced_sampler import BalancedBatchSampler
            balanced_sampler = BalancedBatchSampler(train_dataset, batch_size=config['batch_size'])
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=balanced_sampler,
                num_workers=0,
                pin_memory=False
            )
            print("平衡采样器加载成功")
        except ImportError:
            print("平衡采样器未找到，使用普通采样")
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    # ==================== 结束：数据加载器 ====================

    # 模型与优化器
    print("创建模型...")
    model = UNet().to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # ==================== 损失函数设置 ====================
    print("配置损失函数...")
    
    # 基于真实分布计算类别权重
    class_weights = compute_class_weights_from_distribution().to(device)
    print(f"基于分布的类别权重: {class_weights.cpu().numpy().round(3)}")
    print(f"类别含义: 0=忽略, 1=背景, 2=建筑, 3=道路, 4=水体, 5=荒地, 6=森林, 7=农业")
    # ==================== 结束：损失函数 ====================

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',      # 监控mIoU，要最大化
        factor=0.5,      # 降低因子
        patience=6,      # 6个epoch没改善就降低
        min_lr=1e-6,
        verbose=True
    )

    # 修改所有模型保存路径
    model_dir = os.path.join(BASE_DIR, 'models')

    # 训练准备
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)

    # 记录
    train_losses = []
    val_mious = []
    best_miou = 0
    best_epoch = 0
    no_improve_count = 0
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = os.path.join(BASE_DIR, 'outputs', f'training_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n开始训练 ({config['epochs']}个epoch)...")
    print("=" * 80)

    # 训练循环
    for epoch in range(config['epochs']):
        # ==================== 训练阶段 ====================
        model.train()
        epoch_train_loss = 0
        batch_count = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算组合损失
            loss = compute_combined_loss(
                outputs, masks, class_weights,
                use_focal=config.get('use_focal_loss', True),
                use_boundary=True
            )
            
            # 可选：添加Dice Loss
            if config['loss_weights'].get('dice', 0) > 0:
                loss = loss + config['loss_weights']['dice'] * dice_loss(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            batch_count += 1
            
            # 每几个batch打印一次
            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    batch_iou = compute_iou(preds[0], masks[0])
                    
                    # 计算类别分布
                    unique_classes = torch.unique(masks[0])
                    
                    print(f"Epoch {epoch+1:02d} | Batch {batch_idx+1:03d}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | Batch IoU: {batch_iou:.4f} | "
                          f"类别: {sorted(unique_classes.tolist())}")
        # ==================== 结束：训练阶段 ====================

        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)

        # ==================== 验证阶段 ====================
        model.eval()
        epoch_val_miou = 0
        sample_count = 0
        
        # 各类别IoU统计
        class_iou_sum = torch.zeros(8, device=device)
        class_count = torch.zeros(8, device=device)
        
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                val_preds = torch.argmax(val_outputs, dim=1)
                
                # 计算每个样本的IoU
                for i in range(val_preds.shape[0]):
                    sample_iou = compute_iou(val_preds[i], val_masks[i])
                    epoch_val_miou += sample_iou
                    
                    # 统计各类别IoU
                    for cls in range(8):
                        if cls > 0:  # 跳过忽略区域
                            pred_cls = val_preds[i] == cls
                            mask_cls = val_masks[i] == cls
                            
                            intersection = (pred_cls & mask_cls).sum().item()
                            union = (pred_cls | mask_cls).sum().item()
                            
                            if union > 0:
                                class_iou_sum[cls] += intersection / union
                                class_count[cls] += 1
                
                sample_count += val_preds.shape[0]
        
        avg_val_miou = epoch_val_miou / sample_count
        val_mious.append(avg_val_miou)
        
        # 计算各类别平均IoU
        avg_class_iou = []
        for cls in range(1, 8):  # 跳过忽略区域
            if class_count[cls] > 0:
                avg_class_iou.append((cls, class_iou_sum[cls].item() / class_count[cls].item()))
        # ==================== 结束：验证阶段 ====================

        # 更新学习率
        scheduler.step(avg_val_miou)
        current_lr = optimizer.param_groups[0]['lr']

        # 打印结果
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1:02d}/{config['epochs']} 结果:")
        print(f"{'-'*60}")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证mIoU: {avg_val_miou:.4f} ({avg_val_miou*100:.1f}%)")
        print(f"  学习率: {current_lr:.2e}")
        
        # 显示各类别性能
        if avg_class_iou:
            print(f"\n  各类别IoU (前3):")
            avg_class_iou.sort(key=lambda x: x[1], reverse=True)
            for cls, iou in avg_class_iou[:3]:
                class_names = ['忽略', '背景', '建筑', '道路', '水体', '荒地', '森林', '农业']
                print(f"    {class_names[cls]}({cls}): {iou:.4f}")

        # 显示进步
        if epoch > 0:
            improvement = avg_val_miou - val_mious[-2]
            if improvement > 0:
                print(f"  ↑ mIoU提升: +{improvement:.4f}")
            else:
                print(f"  ↓ mIoU下降: {improvement:.4f}")

        # ==================== 模型保存 ====================
        best_model_path = os.path.join(model_dir, 'unet_best.pth')
        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            best_epoch = epoch + 1
            no_improve_count = 0
            
            # 保存完整检查点
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'train_loss': avg_train_loss,
                'config': config,
                'class_weights': class_weights,
                'class_iou': avg_class_iou,
            }, best_model_path)
            
            print(f"保存最佳模型！mIoU: {best_miou:.4f}")
        else:
            no_improve_count += 1
            print(f"{no_improve_count}个epoch未提升")
        
        # 保存最新模型
        latest_model_path = os.path.join(model_dir, 'unet_latest.pth')
        torch.save(model.state_dict(), latest_model_path)

        # 早停检查
        if no_improve_count >= config['early_stop_patience']:
            print(f"\n早停触发: {no_improve_count}个epoch未提升")
            print(f"   最佳mIoU: {best_miou:.4f} (Epoch {best_epoch})")
            break

        # 定期保存检查点
        checkpoint_path = os.path.join(model_dir, f'unet_epoch{epoch+1}.pth')
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_miou': avg_val_miou,
            }, checkpoint_path)
            print(f"  📁 保存检查点: epoch {epoch+1}")
        
        print(f"{'='*60}\n")

    # ==================== 训练完成 ====================
    print("\n训练完成!")
    print(f"最佳模型在 Epoch {best_epoch}, mIoU: {best_miou:.4f} ({best_miou*100:.1f}%)")

    # 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
    ax1.plot(train_losses, marker='o', linewidth=2, markersize=4)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Loss'], loc='upper right')
        
    ax2.plot(val_mious, marker='s', color='orange', linewidth=2, markersize=4)
    ax2.axhline(y=best_miou, color='r', linestyle='--', linewidth=2, 
                label=f'Best: {best_miou:.3f}')
    ax2.set_title('Validation mIoU', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mIoU', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=120, bbox_inches='tight')
    plt.close()
        
    # 保存训练统计
    stats = {
        'best_miou': float(best_miou),
        'best_epoch': best_epoch,
        'final_train_loss': float(train_losses[-1]),
        'final_val_miou': float(val_mious[-1]),
        'total_epochs_trained': len(train_losses),
        'config': config,
        'class_weights': class_weights.cpu().tolist(),
    }
        
    with open(f'{output_dir}/training_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"\n训练统计:")
    print(f"  开始mIoU: {val_mious[0]:.4f} ({val_mious[0]*100:.1f}%)")
    print(f"  结束mIoU: {val_mious[-1]:.4f} ({val_mious[-1]*100:.1f}%)")
    print(f"  总提升: {(val_mious[-1] - val_mious[0]):.4f} ({(val_mious[-1] - val_mious[0])*100:.1f}%)")
    
    if best_miou >= 0.25:
        print(f"\n良好成绩！mIoU: {best_miou:.4f} ({best_miou*100:.1f}%)")
    elif best_miou >= 0.20:
        print(f"\n中等成绩，还有提升空间: {best_miou:.4f} ({best_miou*100:.1f}%)")
    else:
        print(f"\n需要进一步优化，当前mIoU: {best_miou:.4f} ({best_miou*100:.1f}%)")
        
    print(f"\n训练结果保存在: {output_dir}")
    print(f"最佳模型: {best_model_path}")
    print(f"最新模型: {latest_model_path}")
    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()