#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Monitor Script
Monitors the progress of YOLOv11 training
"""

import os
import time
from pathlib import Path
from datetime import datetime

os.chdir('D:/jglw/yolov11-manhole-detection')

def get_training_progress():
    """Get current training progress from results.csv"""
    results_dir = Path('runs/train/baseline_e50')
    results_csv = results_dir / 'results.csv'

    if not results_csv.exists():
        return None

    try:
        with open(results_csv, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return None

        # Parse header
        header = lines[0].strip().split(',')
        # Get last data line
        last_line = lines[-1].strip().split(',')

        # Create dict
        data = {}
        for i, h in enumerate(header):
            if i < len(last_line):
                try:
                    data[h] = float(last_line[i])
                except:
                    data[h] = last_line[i]

        return data
    except Exception as e:
        print(f"Error reading results: {e}")
        return None

def main():
    print("="*60)
    print("YOLOv11 Training Monitor")
    print("="*60)
    print()

    results_dir = Path('runs/train/baseline_e50')

    if not results_dir.exists():
        print("Training has not started yet.")
        print(f"Expected directory: {results_dir.absolute()}")
        return

    print(f"Monitoring: {results_dir.absolute()}")
    print()

    # Check weights
    weights_dir = results_dir / 'weights'
    if weights_dir.exists():
        print("Saved weights:")
        for w in weights_dir.glob('*.pt'):
            size_mb = w.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(w.stat().st_mtime)
            print(f"  - {w.name} ({size_mb:.2f} MB, modified: {mtime})")
    else:
        print("No weights saved yet.")
    print()

    # Get latest results
    data = get_training_progress()

    if data:
        epoch = int(data.get('epoch', 0)) + 1
        print(f"Current Epoch: {epoch}/50")
        print()
        print("Latest Metrics:")
        print(f"  mAP@0.5:      {data.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP@0.5:0.95: {data.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision:    {data.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall:       {data.get('metrics/recall(B)', 'N/A'):.4f}")
        print()
        print("Training Loss:")
        print(f"  Box Loss:     {data.get('train/box_loss', 'N/A'):.4f}")
        print(f"  Cls Loss:     {data.get('train/cls_loss', 'N/A'):.4f}")
        print(f"  DFL Loss:     {data.get('train/dfl_loss', 'N/A'):.4f}")
        print()
        print("Validation Loss:")
        print(f"  Box Loss:     {data.get('val/box_loss', 'N/A'):.4f}")
        print(f"  Cls Loss:     {data.get('val/cls_loss', 'N/A'):.4f}")
        print(f"  DFL Loss:     {data.get('val/dfl_loss', 'N/A'):.4f}")
    else:
        print("No training results available yet.")
        print("Training may be starting...")

    print()
    print("="*60)

if __name__ == '__main__':
    main()
