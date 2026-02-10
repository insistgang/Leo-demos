@echo off
echo ========================================
echo YOLOv11井盖检测 - 超轻量版训练
echo ========================================
echo.

call conda activate yolov11
cd /d D:\jglw\yolov11-manhole-detection

echo 超轻量参数 (CPU友好):
echo - epochs: 20
echo - batch: 1
echo - image: 320x320 (缩小尺寸)
echo - workers: 0
echo.

python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else \"CPU\"} GB')
print()

from ultralytics import YOLO
print('加载YOLOv11n模型...')
model = YOLO('yolo11n.pt')
print('模型加载完成!')
print()

print('开始训练 (超轻量配置)...')
results = model.train(
    data='configs/data.yaml',
    epochs=20,
    batch=1,
    imgsz=320,
    device='cpu',
    workers=0,
    cache='ram',
    project='runs/train',
    name='baseline_mini',
    exist_ok=True,
    save=True,
    verbose=True
)

print()
print('=== 训练完成！===')
print(f'结果保存在: runs/train/baseline_mini/')
" 2>&1

pause
