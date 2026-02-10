#!/usr/bin/env python3
"""
数据集质量检查脚本
检查井盖数据集的标注质量、类别分布、图像质量等
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DatasetQualityChecker:
    """数据集质量检查器"""

    # 井盖类别定义
    CLASSES = {
        0: "intact",           # 完整
        1: "minor_damaged",    # 轻度破损
        2: "medium_damaged",   # 中度破损
        3: "severe_damaged",   # 重度破损
        4: "missing",          # 缺失
        5: "displaced",        # 移位
        6: "occluded"          # 遮挡
    }

    def __init__(self, image_dir, label_dir=None):
        """
        初始化检查器

        Args:
            image_dir: 图像目录
            label_dir: 标签目录（默认与image_dir同级）
        """
        self.image_dir = Path(image_dir)
        if label_dir is None:
            self.label_dir = self.image_dir.parent / "labels"
        else:
            self.label_dir = Path(label_dir)

        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'images_without_labels': [],
            'labels_without_images': [],
            'corrupted_images': [],
            'invalid_labels': [],
            'class_distribution': defaultdict(int),
            'bbox_sizes': [],
            'bbox_aspect_ratios': [],
            'image_sizes': [],
            'small_objects': [],  # 小目标 (<32x32)
            'large_objects': [],  # 大目标 (>96x96)
        }

    def check_dataset(self):
        """执行完整的数据集检查"""
        print("=" * 60)
        print("数据集质量检查")
        print("=" * 60)
        print(f"图像目录: {self.image_dir}")
        print(f"标签目录: {self.label_dir}")

        # 获取图像文件列表
        image_files = self._get_image_files()
        self.stats['total_images'] = len(image_files)

        print(f"\n找到 {len(image_files)} 张图像")

        if not image_files:
            print("❌ 未找到图像文件")
            return self.stats

        # 检查每张图像
        print("\n开始检查...")

        for img_path in tqdm(image_files, desc="检查进度"):
            self._check_image(img_path)

        # 检查孤立标签文件
        self._check_orphan_labels()

        # 生成报告
        self._generate_report()

        return self.stats

    def _get_image_files(self):
        """获取所有图像文件"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        files = []
        for ext in extensions:
            files.extend(list(self.image_dir.glob(ext)))
        return sorted(files)

    def _check_image(self, img_path):
        """检查单张图像"""
        # 检查图像是否可读
        img = cv2.imread(str(img_path))
        if img is None:
            self.stats['corrupted_images'].append(str(img_path))
            return

        h, w = img.shape[:2]
        self.stats['image_sizes'].append((w, h))

        # 检查对应的标签文件
        label_path = self.label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            self.stats['images_without_labels'].append(str(img_path))
            return

        # 读取标签
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                self.stats['images_without_labels'].append(str(img_path))
                return

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    self.stats['invalid_labels'].append((str(label_path), line))
                    continue

                class_id, x_center, y_center, bbox_w, bbox_h = map(float, parts)
                class_id = int(class_id)

                # 检查类别ID是否有效
                if class_id < 0 or class_id >= len(self.CLASSES):
                    self.stats['invalid_labels'].append((str(label_path), f"Invalid class: {class_id}"))
                    continue

                self.stats['total_labels'] += 1
                self.stats['class_distribution'][class_id] += 1

                # 计算边界框大小（像素）
                abs_w = int(bbox_w * w)
                abs_h = int(bbox_h * h)
                self.stats['bbox_sizes'].append((abs_w, abs_h))

                # 长宽比
                if abs_h > 0:
                    aspect_ratio = abs_w / abs_h
                    self.stats['bbox_aspect_ratios'].append(aspect_ratio)

                # 小目标检测 (<32x32)
                if abs_w < 32 or abs_h < 32:
                    self.stats['small_objects'].append((str(img_path), class_id, abs_w, abs_h))

                # 大目标 (>96x96)
                if abs_w > 96 or abs_h > 96:
                    self.stats['large_objects'].append((str(img_path), class_id, abs_w, abs_h))

        except Exception as e:
            self.stats['invalid_labels'].append((str(label_path), str(e)))

    def _check_orphan_labels(self):
        """检查没有对应图像的标签文件"""
        label_files = list(self.label_dir.glob("*.txt"))

        for label_path in label_files:
            img_path = self.image_dir / f"{label_path.stem}.jpg"
            if not img_path.exists():
                img_path = self.image_dir / f"{label_path.stem}.png"
                if not img_path.exists():
                    self.stats['labels_without_images'].append(str(label_path))

    def _generate_report(self):
        """生成检查报告"""
        print("\n" + "=" * 60)
        print("数据集检查报告")
        print("=" * 60)

        # 基本统计
        print(f"\n【基本统计】")
        print(f"  总图像数: {self.stats['total_images']}")
        print(f"  总标注数: {self.stats['total_labels']}")
        print(f"  平均每张图像标注数: {self.stats['total_labels'] / max(self.stats['total_images'], 1):.2f}")

        # 问题统计
        print(f"\n【问题统计】")
        print(f"  无标签图像: {len(self.stats['images_without_labels'])}")
        print(f"  无图像标签: {len(self.stats['labels_without_images'])}")
        print(f"  损坏图像: {len(self.stats['corrupted_images'])}")
        print(f"  无效标签: {len(self.stats['invalid_labels'])}")

        # 类别分布
        print(f"\n【类别分布】")
        for class_id, count in sorted(self.stats['class_distribution'].items()):
            class_name = self.CLASSES.get(class_id, f"Class_{class_id}")
            percentage = count / max(self.stats['total_labels'], 1) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

        # 边界框大小统计
        if self.stats['bbox_sizes']:
            widths = [w for w, h in self.stats['bbox_sizes']]
            heights = [h for w, h in self.stats['bbox_sizes']]

            print(f"\n【边界框大小】")
            print(f"  宽度 - 最小: {min(widths)}, 最大: {max(widths)}, 平均: {np.mean(widths):.1f}")
            print(f"  高度 - 最小: {min(heights)}, 最大: {max(heights)}, 平均: {np.mean(heights):.1f}")

        # 小目标统计
        print(f"\n【目标大小分布】")
        print(f"  小目标 (<32px): {len(self.stats['small_objects'])} ({len(self.stats['small_objects'])/max(self.stats['total_labels'],1)*100:.1f}%)")
        print(f"  大目标 (>96px): {len(self.stats['large_objects'])} ({len(self.stats['large_objects'])/max(self.stats['total_labels'],1)*100:.1f}%)")

        # 图像尺寸统计
        if self.stats['image_sizes']:
            unique_sizes = Counter(self.stats['image_sizes'])
            print(f"\n【图像尺寸】")
            for size, count in unique_sizes.most_common():
                print(f"  {size[0]}x{size[1]}: {count} 张")

        # 保存详细报告
        self._save_detailed_report()

        # 生成可视化图表
        self._generate_visualizations()

    def _save_detailed_report(self, output_path="data/dataset_quality_report.json"):
        """保存详细的JSON报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'summary': {
                'total_images': self.stats['total_images'],
                'total_labels': self.stats['total_labels'],
                'avg_labels_per_image': self.stats['total_labels'] / max(self.stats['total_images'], 1),
            },
            'issues': {
                'images_without_labels': self.stats['images_without_labels'],
                'labels_without_images': self.stats['labels_without_images'],
                'corrupted_images': self.stats['corrupted_images'],
                'invalid_labels': [f"{file}: {content}" for file, content in self.stats['invalid_labels']],
            },
            'class_distribution': {
                self.CLASSES.get(k, f"Class_{k}"): v
                for k, v in self.stats['class_distribution'].items()
            },
            'small_objects_count': len(self.stats['small_objects']),
            'large_objects_count': len(self.stats['large_objects']),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 详细报告已保存: {output_path}")

    def _generate_visualizations(self, output_dir="results/metrics"):
        """生成可视化图表"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 类别分布柱状图
        if self.stats['class_distribution']:
            plt.figure(figsize=(10, 6))
            classes = [self.CLASSES.get(k, f"Class_{k}") for k in sorted(self.stats['class_distribution'].keys())]
            counts = [self.stats['class_distribution'][k] for k in sorted(self.stats['class_distribution'].keys())]

            plt.bar(classes, counts)
            plt.xlabel('类别')
            plt.ylabel('数量')
            plt.title('类别分布')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'class_distribution.png', dpi=150)
            plt.close()

        # 2. 边界框大小分布
        if self.stats['bbox_sizes']:
            widths = [w for w, h in self.stats['bbox_sizes']]
            heights = [h for w, h in self.stats['bbox_sizes']]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].hist(widths, bins=30, alpha=0.7)
            axes[0].set_xlabel('宽度 (像素)')
            axes[0].set_ylabel('数量')
            axes[0].set_title('边界框宽度分布')

            axes[1].hist(heights, bins=30, alpha=0.7, color='orange')
            axes[1].set_xlabel('高度 (像素)')
            axes[1].set_ylabel('数量')
            axes[1].set_title('边界框高度分布')

            plt.tight_layout()
            plt.savefig(output_dir / 'bbox_size_distribution.png', dpi=150)
            plt.close()

        # 3. 长宽比分布
        if self.stats['bbox_aspect_ratios']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.stats['bbox_aspect_ratios'], bins=50, alpha=0.7)
            plt.xlabel('长宽比 (宽/高)')
            plt.ylabel('数量')
            plt.title('边界框长宽比分布')
            plt.axvline(1.0, color='red', linestyle='--', label='正方形')
            plt.legend()
            plt.savefig(output_dir / 'aspect_ratio_distribution.png', dpi=150)
            plt.close()

        print(f"✅ 可视化图表已保存到: {output_dir}")


def check_all_splits(base_dir):
    """检查所有数据集划分（train/val/test）"""
    base_path = Path(base_dir)

    splits = ['train', 'val', 'test']

    for split in splits:
        image_dir = base_path / "images" / split
        label_dir = base_path / "labels" / split

        if image_dir.exists():
            print(f"\n{'='*60}")
            print(f"检查 {split} 集")
            print(f"{'='*60}")

            checker = DatasetQualityChecker(image_dir, label_dir)
            checker.check_dataset()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='数据集质量检查工具')
    parser.add_argument('--image_dir', type=str, default='data/processed/images/train',
                       help='图像目录')
    parser.add_argument('--label_dir', type=str, default=None,
                       help='标签目录（默认与image_dir同级/labels）')
    parser.add_argument('--check_all', action='store_true',
                       help='检查所有数据集划分（train/val/test）')
    parser.add_argument('--base_dir', type=str, default='data/processed',
                       help='数据集基础目录（用于--check_all）')

    args = parser.parse_args()

    if args.check_all:
        check_all_splits(args.base_dir)
    else:
        checker = DatasetQualityChecker(args.image_dir, args.label_dir)
        checker.check_dataset()


if __name__ == '__main__':
    main()
