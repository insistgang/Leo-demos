#!/usr/bin/env python3
"""
数据预处理脚本
将原始数据集转换为YOLO格式
"""

import argparse
from pathlib import Path
import shutil
import json
from collections import defaultdict
import random

random.seed(42)

def prepare_yolo_dataset(
    raw_dir,
    output_dir="data/processed",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
):
    """
    准备YOLO格式数据集

    Args:
        raw_dir: 原始数据目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """

    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "test").mkdir(parents=True, exist_ok=True)

    print(f"原始数据目录: {raw_path}")
    print(f"输出目录: {output_path}")

    # 假设数据结构:
    # raw/
    #   ├── images/
    #   │   ├── xxx.jpg
    #   │   └── ...
    #   └── labels/
    #       ├── xxx.txt
    #       └── ...

    images_dir = raw_path / "images"
    labels_dir = raw_path / "labels"

    if not images_dir.exists():
        print(f"错误: 图像目录不存在: {images_dir}")
        return

    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(images_dir.glob(ext)))

    if not image_files:
        print(f"警告: 在 {images_dir} 中未找到图像文件")
        return

    print(f"\n找到 {len(image_files)} 张图像")

    # 打乱并划分数据集
    random.shuffle(image_files)
    n = len(image_files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    # 复制文件
    statistics = defaultdict(int)

    for split, files in splits.items():
        print(f"\n处理 {split} 集 ({len(files)} 张)...")

        for img_path in files:
            # 复制图像
            dst_img = output_path / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # 复制标签（如果存在）
            label_name = img_path.stem + '.txt'
            src_label = labels_dir / label_name

            if src_label.exists():
                dst_label = output_path / "labels" / split / label_name
                shutil.copy2(src_label, dst_label)
                statistics[split] += 1
            else:
                print(f"  警告: 未找到标签文件 {label_name}")

    # 统计信息
    print("\n" + "=" * 50)
    print("数据集划分完成")
    print("=" * 50)
    for split in ['train', 'val', 'test']:
        print(f"{split}: {statistics[split]} 张")
    print("=" * 50)

    # 更新data.yaml
    data_yaml_path = Path("configs/data.yaml")
    if data_yaml_path.exists():
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)

        data_yaml['dataset_info'] = {
            'total_images': n,
            'train_size': statistics['train'],
            'val_size': statistics['val'],
            'test_size': statistics['test'],
            'resolution': [640, 640]
        }

        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, allow_unicode=True)

        print(f"\n已更新配置文件: {data_yaml_path}")

def main():
    parser = argparse.ArgumentParser(description='准备YOLO格式数据集')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例')

    args = parser.parse_args()

    prepare_yolo_dataset(
        args.raw_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )

if __name__ == '__main__':
    main()
