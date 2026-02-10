@echo off
REM Roboflow数据集快速下载脚本
REM 使用方法: 1) 先获取API Key 2) 运行此脚本

echo ========================================
echo Roboflow数据集下载工具
echo ========================================
echo.

cd /d D:\jglw\yolov11-manhole-detection

REM 检查是否已激活环境
echo [步骤1/3] 检查conda环境...
conda activate yolov11 2>nul
if errorlevel 1 (
    echo yolov11环境不存在，将创建新环境...
    conda env create -f environment.yml
    conda activate yolov11
)

REM 检查roboflow包
echo.
echo [步骤2/3] 检查roboflow包...
python -c "import roboflow" 2>nul
if errorlevel 1 (
    echo 安装roboflow包...
    pip install roboflow
)

REM 获取API Key
echo.
echo [步骤3/3] 输入Roboflow API Key
echo ----------------------------------------
echo 获取API Key: https://app.roboflow.com/settings/api
echo ----------------------------------------
set /p API_KEY="请输入你的API Key (rf_xxxx): "

if "%API_KEY%"=="" (
    echo 错误: API Key不能为空
    pause
    exit /b 1
)

REM 选择数据集
echo.
echo 可用数据集:
echo [1] sideseeing    - 1,427张, 4类 (推荐)
echo [2] manhole-5k    - 5,000张, 多类
echo [3] road-damage   - 990张, 多类
set /p DATASET_CHOICE="请选择数据集 (1-3): "

if "%DATASET_CHOICE%"=="1" set DATASET=sideseeing
if "%DATASET_CHOICE%"=="2" set DATASET=manhole-5k
if "%DATASET_CHOICE%"=="3" set DATASET=road-damage

if "%DATASET%"=="" (
    echo 错误: 无效选择
    pause
    exit /b 1
)

REM 开始下载
echo.
echo 开始下载数据集: %DATASET%
echo API Key: %API_KEY:~0,10%...
echo.

python scripts/download_roboflow.py --dataset %DATASET% --api-key %API_KEY%

echo.
echo ========================================
echo 下载完成！
echo ========================================
echo.
echo 下一步: 运行数据质量检查
echo python scripts/check_dataset.py --check_all --base_dir data/processed
echo.
pause