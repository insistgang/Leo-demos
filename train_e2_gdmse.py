"""
YOLOv11 + GD-MSE Training Script
Experiment E2: Gradient-Guided Multi-Scale Enhancement
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from modules.gd_mse import GDMSE

# 配置
WORK_DIR = Path("D:/jglw/yolov11-manhole-detection")
DATA_YAML = WORK_DIR / "configs/data.yaml"
OUTPUT_DIR = WORK_DIR / "runs/train/e2_gd_mse"

class YOLOv11_GDMSE(DetectionModel):
    """YOLOv11 with GD-MSE integration"""
    
    def __init__(self, cfg='yolov11n.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        
        # 找到backbone输出层并添加GD-MSE
        self.gdmse = GDMSE(in_channels=256, num_scales=3, use_c3k2_gd=True, use_sppf_gd=False)
        self.gdmse_enabled = True
        
    def forward(self, x, *args, **kwargs):
        """Forward with GD-MSE enhancement"""
        # 标准YOLOv11前向传播
        if not self.gdmse_enabled:
            return super().forward(x, *args, **kwargs)
            
        # 获取中间特征
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            
            # 在backbone结束后应用GD-MSE
            if m.i == 9:  # backbone最后一层
                # 收集多尺度特征
                features = []
                for idx in [6, 8, 9]:  # P3, P4, P5层
                    if y[idx] is not None:
                        features.append(y[idx])
                
                if len(features) == 3:
                    # 应用GD-MSE增强
                    enhanced = self.gdmse(tuple(features))
                    # 更新特征
                    for i, idx in enumerate([6, 8, 9]):
                        y[idx] = enhanced[i]
                    x = enhanced[-1]
        
        return x

def train_yolov11_gdmse():
    """训练YOLOv11 + GD-MSE模型"""
    
    print("="*60)
    print("YOLOv11 + GD-MSE Training")
    print("="*60)
    
    # 检查数据集
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Data config not found: {DATA_YAML}")
    
    print(f"
Data config: {DATA_YAML}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    # 加载基础YOLOv11n模型
    print("
Loading YOLOv11n model...")
    model = YOLO('yolov11n.yaml')
    
    # 注册GD-MSE模块到ultralytics
    from ultralytics.nn import modules
    if not hasattr(modules, 'GDMSE'):
        modules.GDMSE = GDMSE
        print("GD-MSE module registered")
    
    # 训练参数
    train_args = {
        'data': str(DATA_YAML),
        'epochs': 50,
        'batch': 1,
        'imgsz': 320,
        'device': 'cpu',
        'workers': 0,
        'project': str(WORK_DIR / 'runs/train'),
        'name': 'e2_gd_mse',
        'exist_ok': True,
        'pretrained': False,
        'optimizer': 'SGD',
        'lr0': 0.01,
        'verbose': True,
        'patience': 10,
        'save': True,
        'save_period': 10,
    }
    
    print("
Training parameters:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
    
    # 开始训练
    print("
Starting training...")
    try:
        results = model.train(**train_args)
        print("
Training completed successfully!")
        print(f"Results saved to: {OUTPUT_DIR}")
        return results
    except Exception as e:
        print(f"
Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 设置工作目录
    import os
    os.chdir(WORK_DIR)
    
    # 运行训练
    results = train_yolov11_gdmse()
    
    if results:
        print("
" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"Model saved to: {OUTPUT_DIR}")
        print("="*60)
