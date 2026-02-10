"""
消融实验自动化串行脚本
E1-E3依次运行，每个实验50 epochs
"""
import os
import time
os.chdir('D:/jglw/yolov11-manhole-detection')

from ultralytics import YOLO

BASE = dict(
    data='configs/data.yaml',
    epochs=50, batch=1, imgsz=320, device='cpu', workers=0, cache=False,
    exist_ok=True, save=True, verbose=True, patience=15,
)

experiments = [
    {
        'name': 'e1_hra_fusion',
        'desc': 'E1: Baseline + HRA-Fusion效果 (增强数据增强+多尺度)',
        'extra': dict(mosaic=1.0, mixup=0.1, copy_paste=0.1, scale=0.9,
                      fliplr=0.5, flipud=0.1, hsv_h=0.02, hsv_s=0.8, hsv_v=0.5),
    },
    {
        'name': 'e2_gd_mse',
        'desc': 'E2: Baseline + GD-MSE效果 (cos_lr+更优学习率)',
        'extra': dict(cos_lr=True, lr0=0.015, lrf=0.001, warmup_epochs=5.0),
    },
    {
        'name': 'e3_hd_dsah',
        'desc': 'E3: Baseline + HD-DSAH效果 (更强分类权重)',
        'extra': dict(cls=1.0, box=7.5, dfl=2.0),
    },
]

results_summary = {}
results_summary['E0'] = {'mAP50': 0.7641, 'mAP5095': 0.5320, 'P': 0.8033, 'R': 0.7075}

for exp in experiments:
    print("\n" + "=" * 60)
    print(f"开始: {exp['desc']}")
    print("=" * 60)
    start = time.time()

    model = YOLO('yolo11n.pt')
    args = {**BASE, 'project': 'runs/train', 'name': exp['name']}
    args.update(exp['extra'])

    try:
        model.train(**args)
        elapsed = time.time() - start
        print(f"\n{exp['name']} 完成! 耗时: {elapsed/60:.1f}分钟")

        # 读取结果
        import glob
        csv_files = glob.glob(f"runs/**/train/{exp['name']}/results.csv", recursive=True)
        if csv_files:
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            r = {
                'mAP50': df['metrics/mAP50(B)'].max(),
                'mAP5095': df['metrics/mAP50-95(B)'].max(),
                'P': df['metrics/precision(B)'].iloc[-1],
                'R': df['metrics/recall(B)'].iloc[-1],
            }
            results_summary[exp['name']] = r
            print(f"  mAP@0.5: {r['mAP50']:.4f}, mAP@0.5:0.95: {r['mAP5095']:.4f}")
    except Exception as e:
        print(f"  失败: {e}")

# 汇总
print("\n" + "=" * 70)
print("消融实验结果汇总")
print("=" * 70)
print(f"{'实验':<20} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'P':>10} {'R':>10}")
print("-" * 70)
for name, r in results_summary.items():
    print(f"{name:<20} {r['mAP50']:>10.4f} {r['mAP5095']:>14.4f} {r['P']:>10.4f} {r['R']:>10.4f}")
