# 评估模型性能（mIoU等）
# 在验证集上评估训练好的 U-Net 语义分割模型性能，计算 mIoU，
# 并把“预测效果最好 / 中等 / 最差”的样本可视化并保存下来

import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
from datetime import datetime
from datasets.loveda_dataset import SimpleLoveDADataset
from models.unet import UNet

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 获取项目根目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 类别名称
CLASS_NAMES = [
    '忽略区域',    # 0: no-data (ignore)
    '背景',       # 1: background
    '建筑',       # 2: building
    '道路',       # 3: road
    '水体',       # 4: water
    '荒地',       # 5: barren
    '森林',       # 6: forest
    '农业'        # 7: agriculture
]

# 修正颜色映射
CLASS_COLORS = [
    [0, 0, 0],        # 0: 忽略区域 - 黑色
    [128, 128, 128],  # 1: 背景 - 深灰色
    [255, 0, 0],      # 2: 建筑 - 红色
    [255, 255, 0],    # 3: 道路 - 黄色
    [0, 0, 255],      # 4: 水体 - 蓝色
    [165, 42, 42],    # 5: 荒地 - 棕色
    [0, 255, 0],      # 6: 森林 - 绿色
    [255, 165, 0]     # 7: 农业 - 橙色
]

SAVE_TOP_K = 5  # 分别保存最好/中等/最差的样本数量

# 计算所有类别的mIoU
def compute_miou_all_classes(pred, mask, num_classes=8):
    ious = []
    pred_flat = pred.view(-1)
    mask_flat = mask.view(-1)
    
    for cls in range(num_classes):
        pred_i = pred_flat == cls
        mask_i = mask_flat == cls
        
        intersection = (pred_i & mask_i).sum().item()
        union = (pred_i | mask_i).sum().item()
        
        if union == 0:
            # 如果该类别在预测和真实中都不存在，跳过
            continue
        
        ious.append(intersection / union)
    
    return float(np.mean(ious)) if ious else 0.0

# 计算非忽略类别的mIoU
def compute_miou_no_ignore(pred, mask, num_classes=8):
    ious = []
    pred_flat = pred.view(-1)
    mask_flat = mask.view(-1)
    
    for cls in range(1, num_classes):  # 从1开始，跳过忽略区域
        pred_i = pred_flat == cls
        mask_i = mask_flat == cls
        
        intersection = (pred_i & mask_i).sum().item()
        union = (pred_i | mask_i).sum().item()
        
        if union == 0:
            continue
        
        ious.append(intersection / union)
    
    return float(np.mean(ious)) if ious else 0.0

# 计算每个类别的IoU
def compute_class_iou(pred, mask, num_classes=8):
    class_ious = []
    pred_flat = pred.view(-1)
    mask_flat = mask.view(-1)
    
    for cls in range(num_classes):
        pred_i = pred_flat == cls
        mask_i = mask_flat == cls
        
        intersection = (pred_i & mask_i).sum().item()
        union = (pred_i | mask_i).sum().item()
        
        if union > 0:
            class_ious.append(intersection / union)
        else:
            class_ious.append(0.0)
    
    return class_ious

# 可视化函数
def visualize(image, mask, pred, miou_all, miou_no_bg, save_path, sample_idx):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.cpu().numpy()
    pred_np = pred.cpu().numpy()

    # 创建彩色分割图
    true_color = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    pred_color = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
    
    for cls in range(8):
        true_color[mask_np == cls] = CLASS_COLORS[cls]
        pred_color[pred_np == cls] = CLASS_COLORS[cls]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 原始图像
    axes[0].imshow(image_np)
    axes[0].set_title(f"原始图像 #{sample_idx}")
    axes[0].axis('off')
    
    # 真实标签
    axes[1].imshow(true_color)
    axes[1].set_title("真实标签")
    axes[1].axis('off')
    
    # 预测结果
    axes[2].imshow(pred_color)
    axes[2].set_title(f"预测结果\nmIoU(全部): {miou_all:.3f}\nmIoU(非背景): {miou_no_bg:.3f}")
    axes[2].axis('off')
    
    # 类别分布对比
    true_counts = np.bincount(mask_np.flatten(), minlength=8)
    pred_counts = np.bincount(pred_np.flatten(), minlength=8)
    
    x = np.arange(8)
    width = 0.35
    axes[3].bar(x - width/2, true_counts, width, label='真实', color='blue', alpha=0.7)
    axes[3].bar(x + width/2, pred_counts, width, label='预测', color='red', alpha=0.7)
    axes[3].set_xlabel('类别')
    axes[3].set_ylabel('像素数量')
    axes[3].set_title('类别像素分布')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([str(i) for i in range(8)], rotation=45)
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

# 保存类别图例
def save_class_legend(output_dir):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    
    legend_text = "LoveDA类别说明:\n"
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        color_norm = [c/255 for c in color]
        ax.add_patch(plt.Rectangle((0.1, 0.9-i*0.1), 0.05, 0.05, color=color_norm))
        ax.text(0.2, 0.925-i*0.1, f"类别{i}: {name}", fontsize=10, va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_legend.png", dpi=100, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 70)
    print("模型评估")
    print("=" * 70)
    
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 配置
    config = {
        'model_type': 'best',  # 'best'或'latest'
        'data_root': "D:/Projects/SegmentationSystem/2021LoveDA",
        'scene': "both",
        'eval_samples': 50,    # 评估样本数
        'target_size': 256,
        'top_k': SAVE_TOP_K,
    }
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = os.path.join(BASE_DIR, 'outputs', f'evaluation_{timestamp}')
    os.makedirs(os.path.join(output_dir, "best"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mid"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "worst"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)
    
    # 保存类别图例
    save_class_legend(output_dir)
    
    # 加载验证数据
    print("\n加载验证数据...")
    
    dataset = SimpleLoveDADataset(
        data_root=config['data_root'],
        mode="val",
        scene=config['scene'],
        max_samples=config['eval_samples'],
        target_size=config['target_size'],
        augment=False,
        use_offline_resize=True
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"评估数据集: {len(dataset)}个样本")
    
    # 加载模型
    print("\n加载模型...")

    # 选择模型文件
    model_dir = os.path.join(BASE_DIR, 'models')
    
    # 选择模型文件
    if config['model_type'] == 'best':
        model_path = os.path.join(model_dir, 'unet_best.pth')
        print("加载最佳模型 (unet_best.pth)")
    else:
        model_path = os.path.join(model_dir, 'unet_latest.pth')
        print("加载最新模型 (unet_latest.pth)")
    
    model = UNet().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 加载完整检查点
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载完整模型")
            print(f"训练mIoU: {checkpoint.get('best_miou', '未知'):.4f}")
            print(f"训练epoch: {checkpoint.get('epoch', '未知')}")
            
            # 如果有类别权重信息
            if 'class_weights' in checkpoint:
                print(f"   类别权重: {checkpoint['class_weights'].cpu().numpy().round(3)}")
        else:
            # 加载状态字典
            model.load_state_dict(checkpoint)
            print("加载模型权重")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
        # 尝试直接加载
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("直接加载模型权重成功")
        except:
            print("无法加载模型，请检查文件路径")
            return
    
    model.eval()
    
    # 评估循环
    print("\n开始评估...")
    
    results = []
    all_miou_all = []      # 所有类别mIoU
    all_miou_no_bg = []    # 非背景类别mIoU
    
    # 各类别统计
    class_iou_sum = np.zeros(8)
    class_count = np.zeros(8)
    
    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            img = img.to(device)
            mask = mask.to(device)
            
            out = model(img)
            pred = torch.argmax(out, dim=1)
            
            # 计算两种mIoU
            miou_all = compute_miou_all_classes(pred[0], mask[0])
            miou_no_bg = compute_miou_no_ignore(pred[0], mask[0])
            
            # 计算各类别IoU
            class_ious = compute_class_iou(pred[0], mask[0])
            
            # 更新统计
            all_miou_all.append(miou_all)
            all_miou_no_bg.append(miou_no_bg)
            
            for cls in range(8):
                if class_ious[cls] > 0 or (mask[0] == cls).sum().item() > 0:
                    class_iou_sum[cls] += class_ious[cls]
                    if (mask[0] == cls).sum().item() > 0:
                        class_count[cls] += 1
            
            results.append({
                "idx": idx,
                "miou_all": miou_all,
                "miou_no_bg": miou_no_bg,
                "class_ious": class_ious,
                "img": img[0].cpu(),
                "mask": mask[0].cpu(),
                "pred": pred[0].cpu()
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  已评估 {idx + 1}/{len(dataset)} 个样本")
    
    # 计算总体统计
    print("\n计算统计指标...")
    
    # 总体mIoU
    mean_miou_all = np.mean(all_miou_all)
    mean_miou_no_bg = np.mean(all_miou_no_bg)
    
    # 各类别平均IoU
    avg_class_iou = []
    for cls in range(8):
        if class_count[cls] > 0:
            avg_class_iou.append(class_iou_sum[cls] / class_count[cls])
        else:
            avg_class_iou.append(0.0)
    
    # 排序和选择样本
    print("\n选择best/mid/worst样本...")
    
    # 按所有类别mIoU排序
    sorted_results = sorted(results, key=lambda x: x["miou_all"], reverse=True)
    
    # 选择样本
    best_results = sorted_results[:config['top_k']]
    worst_results = sorted_results[-config['top_k']:]
    
    mid_start = len(sorted_results) // 2 - config['top_k'] // 2
    mid_results = sorted_results[mid_start:mid_start + config['top_k']]
    
    # 生成可视化
    print("\n生成可视化...")
    
    # 保存best样本
    for i, r in enumerate(best_results):
        save_path = f"{output_dir}/best/best_{i}_all_{r['miou_all']:.3f}_nobg_{r['miou_no_bg']:.3f}.png"
        visualize(
            r["img"], r["mask"], r["pred"], 
            r["miou_all"], r["miou_no_bg"],
            save_path, r["idx"]
        )
    
    # 保存mid样本
    for i, r in enumerate(mid_results):
        save_path = f"{output_dir}/mid/mid_{i}_all_{r['miou_all']:.3f}_nobg_{r['miou_no_bg']:.3f}.png"
        visualize(
            r["img"], r["mask"], r["pred"],
            r["miou_all"], r["miou_no_bg"],
            save_path, r["idx"]
        )
    
    # 保存worst样本
    for i, r in enumerate(worst_results):
        save_path = f"{output_dir}/worst/worst_{i}_all_{r['miou_all']:.3f}_nobg_{r['miou_no_bg']:.3f}.png"
        visualize(
            r["img"], r["mask"], r["pred"],
            r["miou_all"], r["miou_no_bg"],
            save_path, r["idx"]
        )
    
    # 生成分析图表
    print("\n生成分析图表...")
    
    # 1) 各类别性能条形图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    x = np.arange(8)
    ax1.bar(x, avg_class_iou, color='skyblue')
    ax1.set_xlabel('类别')
    ax1.set_ylabel('IoU')
    ax1.set_title('各类别平均IoU')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i}\n{name}' for i, name in enumerate(CLASS_NAMES[:4])] + 
                       [f'{i}\n{name}' for i, name in enumerate(CLASS_NAMES[4:], start=4)], 
                       rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 2) mIoU分布直方图
    ax2.hist(all_miou_all, bins=15, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(x=mean_miou_all, color='r', linestyle='--', 
                label=f'平均mIoU(全部): {mean_miou_all:.3f}')
    ax2.axvline(x=mean_miou_no_bg, color='b', linestyle=':', 
                label=f'平均mIoU(非背景): {mean_miou_no_bg:.3f}')
    ax2.set_xlabel('mIoU')
    ax2.set_ylabel('样本数量')
    ax2.set_title('mIoU分布直方图')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis/performance_summary.png", dpi=100)
    plt.close()
    
    # 保存详细报告
    print("\n生成评估报告...")
    
    report = {
        '评估配置': config,
        '总体性能': {
            '平均mIoU(全部类别)': float(mean_miou_all),
            '平均mIoU(非背景类别)': float(mean_miou_no_bg),
            '评估样本数': len(results),
            '最佳样本mIoU(全部)': float(best_results[0]['miou_all']),
            '最佳样本mIoU(非背景)': float(best_results[0]['miou_no_bg']),
            '最差样本mIoU(全部)': float(worst_results[-1]['miou_all']),
            '最差样本mIoU(非背景)': float(worst_results[-1]['miou_no_bg']),
        },
        '各类别性能': {},
        '模型信息': {
            '模型路径': model_path,
            '加载类型': config['model_type'],
        }
    }
    
    for cls in range(8):
        report['各类别性能'][f'类别{cls}({CLASS_NAMES[cls]})'] = {
            'IoU': float(avg_class_iou[cls]),
            '出现样本数': int(class_count[cls]),
        }
    
    # 保存JSON报告
    with open(f"{output_dir}/analysis/detailed_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    
    print(f"\n总体性能:")
    print(f"  平均mIoU(全部类别): {mean_miou_all:.4f} ({mean_miou_all*100:.1f}%)")
    print(f"  平均mIoU(非背景): {mean_miou_no_bg:.4f} ({mean_miou_no_bg*100:.1f}%)")
    print(f"  评估样本数: {len(results)}")
    
    print(f"\n各类别IoU排名 (前5):")
    class_perf = []
    for cls in range(8):
        if class_count[cls] > 0:
            class_perf.append((cls, CLASS_NAMES[cls], avg_class_iou[cls], int(class_count[cls])))
    
    class_perf.sort(key=lambda x: x[2], reverse=True)
    
    for i, (cls, name, iou, count) in enumerate(class_perf[:5]):
        print(f"  {i+1}. 类别{cls}({name}): {iou:.4f} (出现{count}次)")
    
    print(f"\n输出目录:")
    print(f"  {output_dir}/")
    print(f"    ├── best/     # 最佳预测样本")
    print(f"    ├── mid/      # 中等预测样本")
    print(f"    ├── worst/    # 最差预测样本")
    print(f"    └── analysis/ # 分析报告和图表")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
