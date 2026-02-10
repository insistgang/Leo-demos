@echo off
echo ========================================
echo YOLOv11井盖检测 - 优化版训练脚本
echo ========================================
echo.

REM 激活conda环境
call conda activate yolov11

REM 进入项目目录
cd /d D:\jglw\yolov11-manhole-detection

echo 当前目录:
cd
echo.

echo 检查Python环境:
python --version
echo.

echo ========================================
echo 开始训练 (CPU优化版)...
echo 参数: batch=8, workers=4
echo ========================================
echo.

python -c "
from ultralytics import YOLO
import os
os.chdir('D:/jglw/yolov11-manhole-detection')
print('开始训练YOLOv11...')
model = YOLO('yolo11n.pt')
results = model.train(
    data='configs/data.yaml',
    epochs=100,
    batch=8,
    imgsz=640,
    device='cpu',
    workers=4,
    project='runs/train',
    name='baseline',
    exist_ok=True,
    save=True,
    verbose=True,
    patience=20
)
print('训练完成!')
print(f'最终mAP@0.5: {results.results_dict}')

echo.
echo ========================================
echo 训练完成！
echo ========================================
pause
"