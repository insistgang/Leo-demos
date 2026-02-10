"""YOLOv11 Baseline训练 - 稳定版"""
import os
os.chdir('D:/jglw/yolov11-manhole-detection')

from ultralytics import YOLO
import torch

print("=" * 60)
print("YOLOv11 井盖检测 - Baseline训练")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"设备: CPU")
print()

print("训练配置:")
print("- 模型: YOLOv11n")
print("- Epochs: 50")
print("- Batch: 1")
print("- Image Size: 320")
print("- Workers: 0")
print("- Cache: False")
print()

model = YOLO('yolo11n.pt')

print("开始训练...")
print("=" * 60)

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
    patience=15
)

print()
print("=" * 60)
print("训练完成！")
print("=" * 60)
print(f"结果保存在: runs/train/baseline_e50/")
