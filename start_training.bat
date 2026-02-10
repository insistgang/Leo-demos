@echo off
echo ========================================
echo YOLOv11井盖检测 - 启动Baseline训练
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

echo 检查PyTorch:
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
echo.

echo 检查数据集:
python scripts/check_dataset.py --base_dir "data/raw/Manhole Cover Dataset YOLO.v2-a002.yolov11" --check_all
echo.

echo ========================================
echo 开始训练...
echo ========================================
echo.

python scripts/train_baseline.py --config configs/baseline.yaml --mode train

echo.
echo ========================================
echo 训练完成！
echo ========================================
pause
