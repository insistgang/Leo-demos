@echo off
chcp 65001 >nul
echo ========================================
echo YOLOv11井盖检测 - Baseline训练启动脚本
echo ========================================
echo.

REM 激活conda环境
echo 激活 yolov11 conda环境...
call conda activate yolov11

REM 切换到项目目录
cd /d D:\jglw\yolov11-manhole-detection

echo 当前目录: %CD%
echo.

REM 检查Python
python --version
echo.

REM 检查模型文件
if not exist yolo11n.pt (
    echo 错误: yolo11n.pt 模型文件不存在！
    pause
    exit /b 1
)

echo 开始训练...
echo 预计时长: 4-5小时
echo.

REM 启动训练（使用start命令在新窗口中运行，避免被意外关闭）
echo 训练窗口将保持打开，直到训练完成或手动关闭...
echo.

python train_baseline_stable.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 训练成功完成!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo 训练未能完成，请检查错误信息
    echo ========================================
)

echo.
echo 按任意键退出...
pause >nul
