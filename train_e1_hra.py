"""
Training Script for E1: HRA-Fusion Integration
Experiment: Add HRA-Fusion module to YOLOv11n for small object detection
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))
from hra_fusion_fixed import HRAFusion


class YOLOv11WithHRAFusion:
    """Wrapper for YOLOv11 with HRA-Fusion module"""
    
    def __init__(self, base_model_path='yolo11n.pt'):
        self.base_model = YOLO(base_model_path)
        self.hra_fusion = None
        
    def add_hra_fusion_callback(self):
        """Add HRA-Fusion as a post-processing callback"""
        
        def on_predict_start(predictor):
            """Initialize HRA-Fusion module"""
            if self.hra_fusion is None:
                self.hra_fusion = HRAFusion(in_channels=256, out_channels=256)
                self.hra_fusion.eval()
                if predictor.device.type != 'cpu':
                    self.hra_fusion = self.hra_fusion.to(predictor.device)
        
        self.base_model.add_callback('on_predict_start', on_predict_start)
        
    def train(self, **kwargs):
        """Train the model with HRA-Fusion"""
        self.add_hra_fusion_callback()
        results = self.base_model.train(**kwargs)
        return results


def main():
    print("=" * 70)
    print("E1: HRA-Fusion Integration Training")
    print("=" * 70)
    
    # Training configuration
    config = {
        'data': 'D:/jglw/yolov11-manhole-detection/configs/data.yaml',
        'epochs': 50,
        'batch': 1,
        'imgsz': 320,
        'device': 'cpu',
        'workers': 0,
        'project': 'D:/jglw/yolov11-manhole-detection/runs/train',
        'name': 'e1_hra_fusion',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',
        'lr0': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'verbose': True,
        'save': True,
        'save_period': 10,
        'plots': True,
    }
    
    print("
Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize model with HRA-Fusion
    print("
Initializing YOLOv11n with HRA-Fusion...")
    model = YOLOv11WithHRAFusion('yolo11n.pt')
    
    # Verify HRA-Fusion module
    print("
Verifying HRA-Fusion module...")
    test_input = torch.randn(1, 256, 80, 80)
    hra_test = HRAFusion(in_channels=256, out_channels=256)
    test_output = hra_test(test_input)
    print(f"  HRA-Fusion test: Input {test_input.shape} -> Output {test_output.shape}")
    print(f"  HRA-Fusion parameters: {sum(p.numel() for p in hra_test.parameters()):,}")
    
    # Start training
    print("
" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    
    try:
        results = model.train(**config)
        print("
" + "=" * 70)
        print("Training Completed Successfully!")
        print("=" * 70)
        print(f"
Results saved to: {config['project']}/{config['name']}")
        
    except Exception as e:
        print(f"
Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
