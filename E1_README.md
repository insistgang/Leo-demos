# E1: HRA-Fusion Integration Experiment

## Overview
This experiment integrates the HRA-Fusion (High-Resolution Adaptive Fusion) module into YOLOv11n for improved small object detection.

## Files Created

### 1. modules/hra_fusion_fixed.py
Fixed version of HRA-Fusion module with memory-efficient implementation:
- Replaced heavy self-attention with LightweightTransformer (convolution-based)
- Maintains dual-branch architecture (CNN local + Global features)
- Includes CBAM attention mechanism
- Parameters: ~491K

### 2. train_e1_hra.py
Training script for E1 experiment:
- Loads YOLOv11n base model
- Integrates HRA-Fusion via callback mechanism
- Uses same training parameters as baseline (epochs=50, batch=1, imgsz=320)
- Saves results to runs/train/e1_hra_fusion

## Training Configuration

```python
epochs: 50
batch: 1
imgsz: 320
device: cpu
workers: 0
optimizer: SGD
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
```

## How to Run

### Verify Setup
```bash
python -c "import sys; sys.path.insert(0, 'modules'); from hra_fusion_fixed import HRAFusion; print('OK')"
```

### Start Training
```bash
python train_e1_hra.py
```

### Expected Output Location
```
runs/train/e1_hra_fusion/
  ├── weights/
  │   ├── best.pt
  │   └── last.pt
  ├── results.csv
  └── plots/
```

## Module Architecture

### HRA-Fusion Module
```
Input (B, 256, H, W)
    ├── Branch A (Local Features)
    │   ├── DepthwiseSeparableConv 3x3
    │   ├── DepthwiseSeparableConv 5x5
    │   └── CBAM Attention
    │
    ├── Branch B (Global Features)
    │   └── LightweightTransformer
    │
    └── Adaptive Fusion
        ├── Weight Calculation (α, β)
        ├── Cross-scale Fusion (P3, P4, P5)
        └── Output Projection
```

## Key Improvements Over Original

1. **Memory Efficiency**: Replaced MultiHeadSelfAttention with LightweightTransformer
   - Original: O(N²) memory complexity
   - Fixed: O(N) memory complexity with convolutions

2. **CPU Compatibility**: All operations optimized for CPU execution
   - No CUDA-specific operations
   - Efficient batch normalization and convolutions

3. **Integration Method**: Uses callback mechanism instead of modifying YOLO architecture
   - Non-invasive integration
   - Easy to enable/disable
   - Compatible with ultralytics framework

## Verification Results

```
[OK] train_e1_hra.py exists
[OK] modules/hra_fusion_fixed.py exists
[OK] configs/data.yaml exists
[OK] HRAFusion module imported
[OK] PyTorch imported
[OK] Ultralytics YOLO imported
[OK] HRA-Fusion forward pass: torch.Size([1, 256, 80, 80]) -> torch.Size([1, 256, 80, 80])
[OK] YOLO model loaded
```

## Notes

- Training on CPU will be slow (estimated 10-20 hours for 50 epochs)
- HRA-Fusion adds ~491K parameters to the base model
- The module is designed for P2 layer (1/4 downsampling) to preserve small object features
- Cross-scale fusion weights: P3(0.3), P4(0.15), P5(0.05)

## Next Steps

1. Run training: `python train_e1_hra.py`
2. Monitor results in runs/train/e1_hra_fusion/results.csv
3. Compare with baseline results from runs/detect/runs/train/baseline_e50/
4. Evaluate mAP improvements on small objects

## Troubleshooting

If training fails:
1. Check data.yaml path is correct
2. Verify yolo11n.pt exists in current directory
3. Ensure sufficient disk space for checkpoints
4. Monitor CPU/memory usage during training
