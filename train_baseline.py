#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv11 Baseline Training
Simple and stable training script
"""

import os
import sys
from pathlib import Path

# Set working directory
os.chdir('D:/jglw/yolov11-manhole-detection')

print("="*60)
print("YOLOv11 Manhole Detection - Baseline Training")
print("="*60)

# Import
from ultralytics import YOLO
import torch

print(f"PyTorch: {torch.__version__}")
print(f"Device: CPU")
print()

print("Training Configuration:")
print("- Model: YOLOv11n")
print("- Epochs: 50")
print("- Batch: 1")
print("- Image Size: 320")
print("- Workers: 0")
print("- Cache: False")
print("- Patience: 15")
print()

# Load model
print("Loading pre-trained model...")
model = YOLO('yolo11n.pt')
print("Model loaded successfully!")
print()

# Start training
print("="*60)
print("Starting training...")
print("="*60)

results = model.train(
    data='configs/data.yaml',
    epochs=50,
    batch=1,
    imgsz=320,
    device='cpu',
    workers=0,
    cache=False,
    project='runs/train',
    name='baseline_e50',
    exist_ok=True,
    save=True,
    verbose=True,
    patience=15,
    amp=False,
    fraction=1.0,
    profile=False,
    plots=True,
    save_period=10,
)

print()
print("="*60)
print("Training completed!")
print("="*60)
print(f"Results saved to: runs/train/baseline_e50/")

# Check weights
weights_dir = Path('runs/train/baseline_e50/weights')
if weights_dir.exists():
    best_pt = weights_dir / 'best.pt'
    last_pt = weights_dir / 'last.pt'

    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"Best weights: {best_pt.absolute()} ({size_mb:.2f} MB)")

    if last_pt.exists():
        size_mb = last_pt.stat().st_size / (1024 * 1024)
        print(f"Last weights: {last_pt.absolute()} ({size_mb:.2f} MB)")
