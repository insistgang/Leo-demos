@echo off
REM YOLOv11井盖检测项目 - Windows快速启动脚本
REM 日期: 2026-02-08

echo =========================================
echo YOLOv11井盖检测项目 - 快速启动
echo =========================================
echo.

REM 项目路径
set PROJECT_DIR=D:\jglw\yolov11-manhole-detection
cd /d %PROJECT_DIR%

REM 步骤1: 检查Python
echo [步骤1/5] 检查Python环境
python --version
if %errorlevel% neq 0 (
    echo [错误] 未找到Python，请先安装Python 3.10+
    pause
    exit /b 1
)
echo [OK] Python已安装
echo.

REM 步骤2: 检查conda环境
echo [步骤2/5] 检查conda环境
conda env list | findstr "^yolov11 "
if %errorlevel% equ 0 (
    echo [OK] yolov11环境已存在
) else (
    echo 创建yolov11环境...
    conda env create -f environment.yml
    echo [OK] 环境创建完成
)
echo.

REM 步骤3: 激活环境并验证
echo [步骤3/5] 激活环境并验证PyTorch
call conda activate yolov11
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo.

REM 步骤4: 检查数据集
echo [步骤4/5] 检查数据集状态
if exist "data\processed\images\train" (
    dir /b data\processed\images\train\*.jpg 2>nul | find /c /v "" > temp.txt
    set /p count=<temp.txt
    del temp.txt
    echo [OK] 训练集图像: %count% 张
) else (
    echo [警告] 数据集未找到，请运行: python scripts\download_modelscope.py
)
echo.

REM 步骤5: 显示下一步操作
echo [步骤5/5] 下一步操作
echo.
echo 环境配置完成！请选择下一步：
echo.
echo 1. 下载数据集:
echo    python scripts\download_modelscope.py
echo.
echo 2. 检查数据质量:
echo    python scripts\check_dataset.py --check_all --base_dir data\processed
echo.
echo 3. 启动baseline训练:
echo    python scripts\train_baseline.py --config configs\baseline.yaml --mode train
echo.
echo 4. 测试YOLOv11推理:
echo    yolo detect predict model=yolo11n.pt source=0
echo.
echo =========================================
echo 快速启动脚本执行完成！
echo =========================================
pause
