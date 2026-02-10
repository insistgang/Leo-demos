"""
Training script for YOLOv11 with HD-DSAH detection head
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from modules.hd_dsah import HDDSAH


class HDDSAHDetectWrapper(nn.Module):
    """Wrapper to make HD-DSAH compatible with YOLO Detect interface"""
    def __init__(self, nc=7, ch=(64, 128, 256)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        
        self.heads = nn.ModuleList([
            HDDSAH(in_channels=c, num_classes=nc, hidden_channels=min(c, 128))
            for c in ch
        ])
        
    def forward(self, x):
        outputs = []
        for i, feat in enumerate(x):
            head_out = self.heads[i](feat, return_hierarchical=False)
            
            cls_pred = head_out["cls"]
            reg_pred = head_out["reg"]
            obj_pred = head_out["obj"]
            
            b, _, h, w = cls_pred.shape
            
            cls_with_obj = torch.cat([cls_pred, obj_pred], dim=1)
            cls_with_obj = cls_with_obj.permute(0, 2, 3, 1).reshape(b, h*w, -1)
            
            reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(b, h*w, 4)
            
            pred = torch.cat([reg_pred, cls_with_obj], dim=-1)
            outputs.append(pred)
        
        if self.training:
            return outputs
        else:
            return torch.cat([o.view(o.shape[0], -1, o.shape[-1]) for o in outputs], dim=1)


def replace_detect_head(model, num_classes=7):
    ch = []
    for i in [16, 19, 22]:
        layer = model.model.model[i]
        if hasattr(layer, 'cv2'):
            ch.append(layer.cv2.conv.out_channels)
        else:
            ch.append(64)
    
    print('Detected input channels:', ch)
    
    new_head = HDDSAHDetectWrapper(nc=num_classes, ch=tuple(ch))
    
    if hasattr(model.model.model[-1], 'stride'):
        new_head.stride = model.model.model[-1].stride
    
    model.model.model[-1] = new_head
    model.model.nc = num_classes
    
    return model


def main():
    print('=' * 60)
    print('YOLOv11 + HD-DSAH Training')
    print('=' * 60)
    
    data_yaml = 'D:/jglw/yolov11-manhole-detection/configs/data.yaml'
    epochs = 50
    batch_size = 1
    imgsz = 320
    device = 'cpu'
    workers = 0
    project = 'D:/jglw/yolov11-manhole-detection/runs/train'
    name = 'e3_hd_dsah'
    
    print()
    print('Configuration:')
    print('  Data:', data_yaml)
    print('  Epochs:', epochs)
    print('  Batch size:', batch_size)
    print('  Image size:', imgsz)
    print('  Device:', device)
    print('  Output:', project + '/' + name)
    
    print()
    print('[1/4] Loading YOLOv11n model...')
    model = YOLO('yolo11n.pt')
    print('  Original head:', type(model.model.model[-1]).__name__)
    
    print()
    print('[2/4] Replacing Detect head with HD-DSAH...')
    model = replace_detect_head(model, num_classes=7)
    print('  New head:', type(model.model.model[-1]).__name__)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print('  Total parameters:', f'{total_params:,}')
    print('  Trainable parameters:', f'{trainable_params:,}')
    
    print()
    print('[3/4] Verifying model forward pass...')
    try:
        dummy_input = torch.randn(1, 3, imgsz, imgsz)
        with torch.no_grad():
            output = model.model(dummy_input)
        print('  Forward pass successful!')
        if isinstance(output, (list, tuple)):
            print('  Output scales:', len(output))
            for i, o in enumerate(output):
                print('    Scale', str(i) + ':', o.shape)
        else:
            print('  Output shape:', output.shape)
    except Exception as e:
        print('  Warning:', e)
        import traceback
        traceback.print_exc()
    
    print()
    print('[4/4] Starting training...')
    print('-' * 60)
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=False,
            verbose=True,
            patience=10,
            save=True,
            plots=True,
        )
        
        print()
        print('=' * 60)
        print('Training completed successfully!')
        print('=' * 60)
        print()
        print('Results saved to:', project + '/' + name)
        
    except Exception as e:
        print()
        print('Training error:', e)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
