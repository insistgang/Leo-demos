import sys
import torch
import ultralytics
from pathlib import Path

print('yolov11环境测试:')
print(f'  Python: {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  Ultralytics: {ultralytics.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print()

os_path = 'D:/jglw/yolov11-manhole-detection'
model_exists = Path(f'{os_path}/yolo11n.pt').exists()
config_exists = Path(f'{os_path}/configs/data.yaml').exists()
dataset_exists = Path(f'{os_path}/data/raw/Manhole Cover Dataset YOLO.v2-a002.yolov11').exists()

print('文件检查:')
print(f'  yolo11n.pt: {"OK" if model_exists else "MISSING"}')
print(f'  configs/data.yaml: {"OK" if config_exists else "MISSING"}')
print(f'  dataset: {"OK" if dataset_exists else "MISSING"}')
print()

if dataset_exists:
    train_imgs = len(list(Path(f'{os_path}/data/raw/Manhole Cover Dataset YOLO.v2-a002.yolov11/train/images').glob('*.jpg')))
    val_imgs = len(list(Path(f'{os_path}/data/raw/Manhole Cover Dataset YOLO.v2-a002.yolov11/valid/images').glob('*.jpg')))
    print(f'数据集统计:')
    print(f'  训练图像: {train_imgs}')
    print(f'  验证图像: {val_imgs}')

# 简单模型加载测试
print()
print('测试加载模型...')
try:
    os.chdir(os_path)
    model = YOLO('yolo11n.pt')
    print('模型加载成功!')
except Exception as e:
    print(f'模型加载失败: {e}')
