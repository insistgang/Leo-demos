@echo off
echo ========================================
echo YOLOv11井盖检测 - CPU稳定版训练
echo ========================================
echo.

call conda activate yolov11
cd /d D:\jglw\yolov11-manhole-detection

echo 使用保守参数避免内存问题:
echo - batch=4 (减小)
echo - workers=2 (减少)
echo - cache=false (关闭缓存)
echo.

python -c "
from ultralytics import YOLO
import os
os.chdir('D:/jglw/yolov11-manhole-detection')
print('开始训练YOLOv11 (CPU稳定版)...')
model = YOLO('yolo11n.pt')
results = model.train(
    data='configs/data.yaml',
    epochs=50,
    batch=4,
    imgsz=640,
    device='cpu',
    workers=2,
    cache=False,
    project='runs/train',
    name='baseline_cpu',
    exist_ok=True,
    save=True,
    verbose=True,
    patience=10
)
print('训练完成!')
"

pause
