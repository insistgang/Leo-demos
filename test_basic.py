"""基础功能测试"""
import os
os.chdir('D:/jglw/yolov11-manhole-detection')

print("=" * 50)
print("步骤1: 导入库")
print("=" * 50)
from ultralytics import YOLO
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Ultralytics导入成功")

print("\n" + "=" * 50)
print("步骤2: 加载模型")
print("=" * 50)
model = YOLO('yolo11n.pt')
print("✓ 模型加载成功")

print("\n" + "=" * 50)
print("步骤3: 验证数据集配置")
print("=" * 50)
import yaml
with open('configs/data.yaml', 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)
print(f"✓ 数据集路径: {data_config['path']}")
print(f"✓ 类别数: {data_config['nc']}")

print("\n" + "=" * 50)
print("步骤4: 检查训练图像")
print("=" * 50)
train_path = os.path.join(data_config['path'], 'train/images')
if os.path.exists(train_path):
    images = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.png'))]
    print(f"✓ 训练图像数量: {len(images)}")
    if images:
        print(f"✓ 示例图像: {images[0]}")
else:
    print(f"✗ 训练路径不存在: {train_path}")

print("\n" + "=" * 50)
print("步骤5: 尝试训练1个epoch (batch=1, imgsz=320)")
print("=" * 50)
try:
    results = model.train(
        data='configs/data.yaml',
        epochs=1,
        batch=1,
        imgsz=320,
        device='cpu',
        workers=0,
        cache=False,
        project='runs/train',
        name='test_basic',
        exist_ok=True,
        save=False,
        verbose=True
    )
    print("✓ 训练完成！")
except Exception as e:
    print(f"✗ 训练失败: {e}")
    import traceback
    traceback.print_exc()
