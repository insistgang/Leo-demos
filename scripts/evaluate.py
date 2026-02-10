#!/usr/bin/env python3
"""
模型评估脚本
用法: python evaluate.py --model runs/train/baseline/weights/best.pt
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import json

def evaluate_model(model_path, data_config='configs/data.yaml'):
    """评估模型性能"""

    print("=" * 50)
    print("模型评估")
    print("=" * 50)
    print(f"模型: {model_path}")
    print(f"数据: {data_config}")
    print("=" * 50)

    # 加载模型
    model = YOLO(model_path)

    # 验证集评估
    print("\n验证集评估...")
    metrics = model.val(data=data_config, split='val')

    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    # 各类别AP
    print("\n各类别AP@0.5:")
    for i, ap in enumerate(metrics.box.map50_per_class):
        print(f"  类别 {i}: {ap:.4f}")

    # 测试集评估（如果有）
    print("\n测试集评估...")
    test_metrics = model.val(data=data_config, split='test')

    print("\n" + "=" * 50)
    print("测试集结果")
    print("=" * 50)
    print(f"mAP@0.5: {test_metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {test_metrics.box.map:.4f}")

    # 保存结果
    results = {
        'val': {
            'map50': float(metrics.box.map50),
            'map': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        },
        'test': {
            'map50': float(test_metrics.box.map50),
            'map': float(test_metrics.box.map),
        }
    }

    output_path = Path('results/metrics')
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / f'{Path(model_path).stem}_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {output_path}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description='YOLOv11模型评估')
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data', type=str, default='configs/data.yaml',
                        help='数据集配置路径')

    args = parser.parse_args()

    evaluate_model(args.model, args.data)

if __name__ == '__main__':
    main()
