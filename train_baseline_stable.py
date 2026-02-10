#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv11 Baseline训练 - 稳定版
包含错误处理、进度日志、训练恢复等功能
"""

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# 确保工作目录正确
os.chdir('D:/jglw/yolov11-manhole-detection')
sys.path.insert(0, 'D:/jglw/yolov11-manhole-detection')

def setup_logging():
    """设置日志记录"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_baseline_{timestamp}.log'

    return log_file

def log_message(log_file, message):
    """记录日志到文件和标准输出"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"
    print(log_line)

    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')

def check_environment():
    """检查训练环境"""
    print("\n" + "="*60)
    print("环境检查")
    print("="*60)

    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")

    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"训练设备: CPU")
    except Exception as e:
        print(f"PyTorch检查失败: {e}")
        return False

    # 检查Ultralytics
    try:
        import ultralytics
        print(f"Ultralytics版本: {ultralytics.__version__}")
    except Exception as e:
        print(f"Ultralytics检查失败: {e}")
        return False

    # 检查模型文件
    model_path = Path('yolo11n.pt')
    if not model_path.exists():
        print(f"错误: 模型文件 {model_path} 不存在!")
        return False
    print(f"预训练模型: {model_path} (存在)")

    # 检查数据集配置
    data_config = Path('configs/data.yaml')
    if not data_config.exists():
        print(f"错误: 数据配置文件 {data_config} 不存在!")
        return False
    print(f"数据配置: {data_config} (存在)")

    # 检查数据集
    dataset_path = Path('data/raw/Manhole Cover Dataset YOLO.v2-a002.yolov11')
    if not dataset_path.exists():
        print(f"错误: 数据集目录 {dataset_path} 不存在!")
        return False

    train_images = list((dataset_path / 'train' / 'images').glob('*.jpg')) + \
                   list((dataset_path / 'train' / 'images').glob('*.png'))
    val_images = list((dataset_path / 'valid' / 'images').glob('*.jpg')) + \
                 list((dataset_path / 'valid' / 'images').glob('*.png'))

    print(f"训练集图像数: {len(train_images)}")
    print(f"验证集图像数: {len(val_images)}")

    if len(train_images) == 0 or len(val_images) == 0:
        print("警告: 数据集图像数为0，请检查数据集路径!")
        return False

    print("="*60)
    return True

def main():
    """主训练函数"""
    log_file = setup_logging()

    log_message(log_file, "\n" + "="*60)
    log_message(log_file, "YOLOv11 井盖检测 - Baseline训练")
    log_message(log_file, "="*60)

    # 环境检查
    if not check_environment():
        log_message(log_file, "环境检查失败，退出训练!")
        return False

    log_message(log_file, "")
    log_message(log_file, "训练配置:")
    log_message(log_file, "- 模型: YOLOv11n (yolo11n.pt)")
    log_message(log_file, "- Epochs: 50")
    log_message(log_file, "- Batch Size: 1")
    log_message(log_file, "- Image Size: 320")
    log_message(log_file, "- Device: CPU")
    log_message(log_file, "- Workers: 0")
    log_message(log_file, "- Cache: False")
    log_message(log_file, "- Patience: 15")
    log_message(log_file, "- Project: runs/train")
    log_message(log_file, "- Name: baseline_e50")
    log_message(log_file, "")

    # 导入必要的库
    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        log_message(log_file, f"导入库失败: {e}")
        traceback.print_exc()
        return False

    # 加载模型
    log_message(log_file, "加载预训练模型...")
    try:
        model = YOLO('yolo11n.pt')
        log_message(log_file, "模型加载成功!")
    except Exception as e:
        log_message(log_file, f"模型加载失败: {e}")
        traceback.print_exc()
        return False

    # 开始训练
    log_message(log_file, "")
    log_message(log_file, "="*60)
    log_message(log_file, "开始训练...")
    log_message(log_file, "="*60)

    start_time = time.time()

    try:
        results = model.train(
            data='configs/data.yaml',
            epochs=50,
            batch=1,
            imgsz=320,
            device='cpu',
            workers=0,
            cache=False,
            project='runs/train',
            name='baseline_e50',
            exist_ok=True,
            save=True,
            verbose=True,
            patience=15,
            # 额外的稳定参数
            amp=False,           # 禁用混合精度训练
            fraction=1.0,       # 使用全部数据
            profile=False,       # 禁用性能分析
            plots=True,          # 生成训练图表
            save_period=10,      # 每10个epoch保存一次
        )

        end_time = time.time()
        training_duration = end_time - start_time

        log_message(log_file, "")
        log_message(log_file, "="*60)
        log_message(log_file, "训练完成!")
        log_message(log_file, "="*60)
        log_message(log_file, f"训练时长: {training_duration/3600:.2f} 小时")

        # 检查结果文件
        results_dir = Path('runs/train/baseline_e50')
        weights_dir = results_dir / 'weights'

        if weights_dir.exists():
            best_pt = weights_dir / 'best.pt'
            last_pt = weights_dir / 'last.pt'

            if best_pt.exists():
                size_mb = best_pt.stat().st_size / (1024 * 1024)
                log_message(log_file, f"最佳权重: {best_pt.absolute()} ({size_mb:.2f} MB)")

            if last_pt.exists():
                size_mb = last_pt.stat().st_size / (1024 * 1024)
                log_message(log_file, "最后权重: {} ({:.2f} MB)".format(last_pt.absolute(), size_mb))

        # 打印最终结果
        log_message(log_file, "")
        log_message(log_file, "训练结果摘要:")
        log_message(log_file, "- 完整结果请查看: runs/train/baseline_e50/")

        # 尝试读取results.csv
        results_csv = results_dir / 'results.csv'
        if results_csv.exists():
            log_message(log_file, f"结果CSV: {results_csv.absolute()}")

        return True

    except KeyboardInterrupt:
        log_message(log_file, "")
        log_message(log_file, "训练被用户中断!")
        return False
    except Exception as e:
        log_message(log_file, "")
        log_message(log_file, f"训练过程出错: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n训练成功完成!")
        sys.exit(0)
    else:
        print("\n训练未能完成，请检查日志!")
        sys.exit(1)
