@echo off
REM YOLOv11 Baseline Training Launcher
REM This script activates the yolov11 conda environment and starts training

echo ========================================
echo YOLOv11 Manhole Detection Training
echo ========================================
echo.
echo Activating yolov11 conda environment...

call D:\anaconda\Scripts\activate.bat yolov11

cd /d D:\jglw\yolov11-manhole-detection

echo Current directory: %CD%
echo.
echo Python version:
python --version
echo.

echo Starting training...
echo Estimated time: 4-5 hours
echo.

python train_baseline.py

echo.
echo ========================================
if %ERRORLEVEL% EQU 0 (
    echo Training completed successfully!
) else (
    echo Training failed or was interrupted.
)
echo ========================================
echo.

pause
