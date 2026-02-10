#!/usr/bin/env python3
"""
YOLOv11井盖检测训练脚本
用法: python train.py --config configs/baseline.yaml
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_model(config):
    """训练模型"""

    # 打印训练信息
    print("=" * 50)
    print("YOLOv11井盖检测训练")
    print("=" * 50)
    print(f"模型: {config.get('model', 'yolo11n.pt')}")
    print(f"数据集: {config.get('data', 'configs/data.yaml')}")
    print(f"Epochs: {config.get('epochs', 100)}")
    print(f"Batch Size: {config.get('batch', 16)}")
    print(f"图像尺寸: {config.get('imgsz', 640)}")
    print("=" * 50)

    # 加载模型
    model = YOLO(config.get('model', 'yolo11n.pt'))

    # 训练参数
    train_args = {
        'data': config.get('data', 'configs/data.yaml'),
        'epochs': config.get('epochs', 100),
        'batch': config.get('batch', 16),
        'imgsz': config.get('imgsz', 640),
        'optimizer': config.get('optimizer', 'SGD'),
        'lr0': config.get('lr0', 0.01),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'device': config.get('device', 0),
        'workers': config.get('workers', 8),
        'project': config.get('project', 'runs/train'),
        'name': config.get('name', 'experiment'),
        'exist_ok': config.get('exist_ok', False),
        'pretrained': config.get('pretrained', True),
        'verbose': config.get('verbose', True),
        'seed': config.get('seed', 42),
        'save': config.get('save', True),
        'cos_lr': config.get('cos_lr', True),
    }

    # 添加数据增强参数
    aug_params = ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate',
                  'scale', 'shear', 'perspective', 'flipud', 'fliplr',
                  'mosaic', 'mixup']
    for param in aug_params:
        if param in config:
            train_args[param] = config[param]

    # 开始训练
    results = model.train(**train_args)

    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print("=" * 50)

    return results

def main():
    parser = argparse.ArgumentParser(description='YOLOv11井盖检测训练')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                        help='配置文件路径')
    parser.add_argument('--data', type=str, default=None,
                        help='数据集配置路径（覆盖配置文件）')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch', type=int, default=None,
                        help='批次大小（覆盖配置文件）')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置文件
    if args.data:
        config['data'] = args.data
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch:
        config['batch'] = args.batch

    # 训练模型
    train_model(config)

if __name__ == '__main__':
    main()
