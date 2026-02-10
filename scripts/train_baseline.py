#!/usr/bin/env python3
"""
YOLOv11 Baseline训练脚本
支持完整训练流程：数据验证、训练、评估、可视化
"""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class YOLOv11Trainer:
    """YOLOv11训练器"""

    def __init__(self, config_path="configs/baseline.yaml"):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.experiment_dir = None
        self.results = None

    def _load_config(self, config_path):
        """加载配置文件"""
        config_path = Path(config_path)
        if not config_path.exists():
            # 默认配置
            return {
                'model': 'yolo11n.pt',
                'data': 'configs/data.yaml',
                'epochs': 100,
                'batch': 16,
                'imgsz': 640,
                'device': 0,
                'workers': 8,
                'project': 'runs/train',
                'name': 'baseline',
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'cos_lr': True,
                'mosaic': 1.0,
                'mixup': 0.0,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'fliplr': 0.5,
                'flipud': 0.0,
                'paste_in': 0.0,
                'seed': 42,
            }

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def validate_data(self):
        """验证数据集"""
        print("=" * 60)
        print("数据集验证")
        print("=" * 60)

        data_config = self.config.get('data', 'configs/data.yaml')
        data_path = Path(data_config)

        if not data_path.exists():
            print(f"❌ 数据配置文件不存在: {data_path}")
            return False

        # 加载数据配置
        with open(data_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)

        print(f"\n数据集路径: {data_yaml.get('path')}")
        print(f"训练集: {data_yaml.get('train')}")
        print(f"验证集: {data_yaml.get('val')}")
        print(f"测试集: {data_yaml.get('val')}")

        # 统计类别
        names = data_yaml.get('names', {})
        nc = data_yaml.get('nc', len(names))

        print(f"\n类别数量: {nc}")
        print("类别列表:")
        for idx, name in names.items():
            print(f"  {idx}: {name}")

        # 检查图像和标签
        base_path = Path(data_yaml.get('path', 'data/processed'))

        for split in ['train', 'val']:
            img_dir = base_path / data_yaml.get(f'{split}', f'images/{split}')
            label_dir = base_path / data_yaml.get(f'{split}', f'labels/{split}')

            img_count = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
            label_count = len(list(label_dir.glob('*.txt'))) if label_dir.exists() else 0

            print(f"\n{split}集:")
            print(f"  图像: {img_count} 张")
            print(f"  标签: {label_count} 个")

        print("\n✅ 数据集验证完成")
        return True

    def create_experiment_dir(self):
        """创建实验目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config.get('name', 'experiment')}_{timestamp}"

        self.experiment_dir = Path(self.config.get('project', 'runs/train')) / exp_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.experiment_dir / 'weights').mkdir(exist_ok=True)
        (self.experiment_dir / 'logs').mkdir(exist_ok=True)
        (self.experiment_dir / 'results').mkdir(exist_ok=True)
        (self.experiment_dir / 'plots').mkdir(exist_ok=True)

        # 保存实验配置
        config_path = self.experiment_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)

        print(f"实验目录: {self.experiment_dir}")
        return self.experiment_dir

    def train(self):
        """执行训练"""
        print("\n" + "=" * 60)
        print("YOLOv11 Baseline 训练")
        print("=" * 60)

        # 打印训练配置
        print(f"\n模型: {self.config.get('model')}")
        print(f"数据集: {self.config.get('data')}")
        print(f"Epochs: {self.config.get('epochs')}")
        print(f"Batch Size: {self.config.get('batch')}")
        print(f"图像尺寸: {self.config.get('imgsz')}")
        print(f"设备: {'CUDA' if self.config.get('device') == '0' else 'CPU'}")
        print(f"优化器: {self.config.get('optimizer')}")
        print(f"初始学习率: {self.config.get('lr0')}")
        print(f"权重衰减: {self.config.get('weight_decay')}")

        # 创建实验目录
        self.create_experiment_dir()

        # 加载模型
        model = YOLO(self.config.get('model'))

        # 构建训练参数
        train_args = {
            'data': self.config.get('data'),
            'epochs': self.config.get('epochs', 100),
            'batch': self.config.get('batch', 16),
            'imgsz': self.config.get('imgsz', 640),
            'device': self.config.get('device', 0),
            'workers': self.config.get('workers', 8),
            'project': str(Path(self.config.get('project', 'runs/train'))),
            'name': self.experiment_dir.name,
            'exist_ok': True,
            'pretrained': self.config.get('pretrained', True),
            'optimizer': self.config.get('optimizer', 'AdamW'),
            'lr0': self.config.get('lr0', 0.001),
            'lrf': self.config.get('lrf', 0.01),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'warmup_epochs': self.config.get('warmup_epochs', 3),
            'cos_lr': self.config.get('cos_lr', True),
            'seed': self.config.get('seed', 42),
            'verbose': True,
            'save': True,
            'save_period': 10,
            'plots': True,
        }

        # 数据增强参数
        aug_params = [
            'mosaic', 'mixup', 'hsv_h', 'hsv_s', 'hsv_v',
            'degrees', 'translate', 'scale', 'fliplr', 'flipud',
            'paste_in', 'copy_paste'
        ]
        for param in aug_params:
            if param in self.config:
                train_args[param] = self.config[param]

        # 开始训练
        print("\n" + "=" * 60)
        print("开始训练...")
        print("=" * 60)

        self.results = model.train(**train_args)

        # 保存训练结果摘要
        self._save_training_summary()

        return self.results

    def _save_training_summary(self):
        """保存训练摘要"""
        if self.results is None or self.experiment_dir is None:
            return

        summary = {
            'experiment_name': self.experiment_dir.name,
            'model': self.config.get('model'),
            'epochs_completed': self.config.get('epochs', 100),
            'final_results': {
                'map50': float(self.results.box.map50),
                'map': float(self.results.box.map),
                'precision': float(self.results.box.mp),
                'recall': float(self.results.box.mr),
            },
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        summary_path = self.experiment_dir / 'training_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n训练摘要已保存: {summary_path}")

    def evaluate(self, model_path=None, split='val'):
        """
        评估模型

        Args:
            model_path: 模型权重路径，默认使用训练好的best.pt
            split: 评估集划分 (val/test)
        """
        if model_path is None:
            model_path = self.experiment_dir / 'weights' / 'best.pt'

        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return None

        print(f"\n评估模型: {model_path}")
        print(f"数据集划分: {split}")

        model = YOLO(str(model_path))
        metrics = model.val(data=self.config.get('data'), split=split)

        # 打印结果
        print("\n" + "=" * 60)
        print(f"{split}集评估结果")
        print("=" * 60)
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"mAP@0.75: {metrics.box.map75:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")

        # 各类别AP
        print("\n各类别AP@0.5:")
        for i, ap in enumerate(metrics.box.map50_per_class):
            name = self._get_class_name(i)
            if ap is not None:
                print(f"  {name}: {ap:.4f}")

        # 保存评估结果
        self._save_evaluation_results(metrics, split)

        return metrics

    def _get_class_name(self, class_id):
        """获取类别名称"""
        data_config = self.config.get('data', 'configs/data.yaml')
        try:
            with open(data_config, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
            names = data_yaml.get('names', {})
            return names.get(str(class_id), f"Class_{class_id}")
        except:
            return f"Class_{class_id}"

    def _save_evaluation_results(self, metrics, split):
        """保存评估结果"""
        results = {
            'split': split,
            'map50': float(metrics.box.map50),
            'map': float(metrics.box.map),
            'map75': float(metrics.box.box.map75) if hasattr(metrics.box, 'map75') else 0,
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'per_class_ap50': {
                self._get_class_name(i): float(ap) if ap is not None else None
                for i, ap in enumerate(metrics.box.map50_per_class)
            },
            'timestamp': datetime.now().isoformat()
        }

        output_path = self.experiment_dir / 'results' / f'{split}_results.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"评估结果已保存: {output_path}")

    def plot_training_curves(self, csv_path=None):
        """
        绘制训练曲线

        Args:
            csv_path: results.csv路径，默认自动查找
        """
        if csv_path is None:
            # 查找results.csv
            csv_path = self.experiment_dir / 'results.csv'

        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"❌ 训练结果文件不存在: {csv_path}")
            return

        # 读取CSV
        df = pd.read_csv(csv_path)

        # 绘制训练曲线
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss曲线
        if 'train/loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train/loss'], label='Train Loss', marker='o')
            if 'val/loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['val/loss'], label='Val Loss', marker='s')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # mAP曲线
        if 'metrics/mAP50(B)' in df.columns:
            axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', marker='o', color='green')
            if 'metrics/mAP50-95(B)' in df.columns:
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', marker='s', color='blue')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].set_title('Mean Average Precision')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # Precision和Recall
        if 'metrics/precision(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o', color='orange')
        if 'metrics/recall(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='s', color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 学习率曲线
        if 'train/lr0' in df.columns or 'lr' in df.columns:
            lr_col = 'train/lr0' if 'train/lr0' in df.columns else 'lr'
            axes[1, 1].plot(df['epoch'], df[lr_col], marker='o', color='purple')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)

        plt.tight_layout()

        # 保存图像
        plot_path = self.experiment_dir / 'plots' / 'training_curves.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"训练曲线已保存: {plot_path}")

    def generate_inference_demo(self, num_samples=5):
        """生成推理演示"""
        model_path = self.experiment_dir / 'weights' / 'best.pt'
        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return

        model = YOLO(str(model_path))

        # 获取测试图像
        data_config = self.config.get('data', 'configs/data.yaml')
        with open(data_config, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)

        base_path = Path(data_yaml.get('path', 'data/processed'))
        test_dir = base_path / 'images' / 'test'

        if not test_dir.exists():
            test_dir = base_path / 'images' / 'val'

        test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        test_images = test_images[:num_samples]

        if not test_images:
            print("❌ 未找到测试图像")
            return

        # 运行推理
        results_dir = self.experiment_dir / 'results' / 'inference'
        results_dir.mkdir(parents=True, exist_ok=True)

        for img_path in test_images:
            results = model(str(img_path), save=True, project=str(results_dir), name=img_path.stem, conf=0.25)

        print(f"推理结果已保存到: {results_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv11 Baseline训练工具')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'plot', 'demo'],
                       help='运行模式: train-训练, eval-评估, plot-绘制曲线, demo-推理演示')
    parser.add_argument('--model', type=str, default=None,
                       help='模型权重路径 (用于eval模式)')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='评估数据集划分')

    args = parser.parse_args()

    # 创建训练器
    trainer = YOLOv11Trainer(args.config)

    if args.mode == 'train':
        # 验证数据
        if not trainer.validate_data():
            print("❌ 数据验证失败，请检查数据集配置")
            return

        # 训练
        trainer.train()

        # 训练后评估
        trainer.evaluate(split='val')

        # 绘制曲线
        trainer.plot_training_curves()

        # 生成推理演示
        trainer.generate_inference_demo()

    elif args.mode == 'eval':
        # 仅评估
        trainer.evaluate(model_path=args.model, split=args.split)

    elif args.mode == 'plot':
        # 仅绘制曲线
        trainer.plot_training_curves()

    elif args.mode == 'demo':
        # 仅推理演示
        trainer.generate_inference_demo()

    print("\n✅ 任务完成!")


if __name__ == '__main__':
    main()
