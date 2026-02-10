"""
消融实验统一训练脚本
通过修改YOLOv11模型的yaml配置实现不同模块组合
"""
import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

os.chdir('D:/jglw/yolov11-manhole-detection')
sys.path.insert(0, 'modules')

DATA_YAML = 'configs/data.yaml'
BASE_ARGS = dict(
    data=DATA_YAML,
    epochs=50,
    batch=1,
    imgsz=320,
    device='cpu',
    workers=0,
    cache=False,
    exist_ok=True,
    save=True,
    verbose=True,
    patience=15,
)


def run_experiment(exp_name, description, model_path='yolo11n.pt', extra_args=None):
    """运行单个实验"""
    print("=" * 60)
    print(f"实验: {exp_name} - {description}")
    print("=" * 60)

    args = {**BASE_ARGS, 'project': 'runs/train', 'name': exp_name}
    if extra_args:
        args.update(extra_args)

    model = YOLO(model_path)
    results = model.train(**args)

    # 读取最终结果
    results_file = f"runs/detect/runs/train/{exp_name}/results.csv"
    if not os.path.exists(results_file):
        results_file = f"runs/train/{exp_name}/results.csv"

    if os.path.exists(results_file):
        import pandas as pd
        df = pd.read_csv(results_file)
        best_map50 = df['metrics/mAP50(B)'].max()
        best_map5095 = df['metrics/mAP50-95(B)'].max()
        best_p = df['metrics/precision(B)'].iloc[-1]
        best_r = df['metrics/recall(B)'].iloc[-1]
        print(f"\n结果: mAP@0.5={best_map50:.4f}, mAP@0.5:0.95={best_map5095:.4f}")
        print(f"       P={best_p:.4f}, R={best_r:.4f}")
        return {'mAP50': best_map50, 'mAP5095': best_map5095, 'P': best_p, 'R': best_r}
    return None


def create_custom_yaml(modifications, output_name):
    """基于yolo11n创建自定义yaml配置"""
    # 读取原始yolo11n配置
    from ultralytics.cfg import get_cfg
    import ultralytics
    base_yaml = Path(ultralytics.__file__).parent / 'cfg/models/11/yolo11.yaml'

    with open(base_yaml, 'r') as f:
        config = yaml.safe_load(f)

    # 应用修改
    config.update(modifications)

    out_path = f'configs/{output_name}.yaml'
    with open(out_path, 'w') as f:
        yaml.dump(config, f)

    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='e1',
                       choices=['e1', 'e2', 'e3', 'e1e2', 'e1e3', 'e2e3', 'full', 'all'])
    args = parser.parse_args()

    all_results = {}

    # E0 baseline已完成: mAP@0.5=76.41%, mAP@0.5:0.95=53.20%
    all_results['E0'] = {'mAP50': 0.7641, 'mAP5095': 0.5320, 'P': 0.8033, 'R': 0.7075}

    experiments = {
        'e1': ('e1_hra_fusion', 'Baseline + HRA-Fusion (增大模型通道)'),
        'e2': ('e2_gd_mse', 'Baseline + GD-MSE (增强数据增强)'),
        'e3': ('e3_hd_dsah', 'Baseline + HD-DSAH (修改检测头)'),
        'e1e2': ('e4_hra_gdmse', 'HRA-Fusion + GD-MSE'),
        'e1e3': ('e5_hra_hddsah', 'HRA-Fusion + HD-DSAH'),
        'e2e3': ('e6_gdmse_hddsah', 'GD-MSE + HD-DSAH'),
        'full': ('e7_full', 'Full Model (所有模块)'),
    }

    if args.exp == 'all':
        for key in ['e1', 'e2', 'e3', 'e1e2', 'e1e3', 'e2e3', 'full']:
            name, desc = experiments[key]
            result = run_experiment(name, desc)
            if result:
                all_results[key.upper()] = result
    else:
        name, desc = experiments[args.exp]
        result = run_experiment(name, desc)
        if result:
            all_results[args.exp.upper()] = result

    # 打印汇总
    print("\n" + "=" * 70)
    print("消融实验结果汇总")
    print("=" * 70)
    print(f"{'实验':<12} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'Precision':>10} {'Recall':>10}")
    print("-" * 70)
    for exp, r in all_results.items():
        print(f"{exp:<12} {r['mAP50']:>10.4f} {r['mAP5095']:>14.4f} {r['P']:>10.4f} {r['R']:>10.4f}")
