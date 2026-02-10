# YOLOv11井盖状态检测系统

## 项目结构

```
yolov11-manhole-detection/
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   │   ├── images/            # 图像 (train/val/test)
│   │   └── labels/            # 标签 (train/val/test)
│   └── annotations/           # 标注文件
├── models/                    # 模型目录
│   ├── baseline/              # 基线模型
│   ├── improved/              # 改进模型
│   │   ├── hra_fusion/        # HRA-Fusion模块
│   │   ├── gd_mse/            # GD-MSE模块
│   │   └── hd_dsah/           # HD-DSAH检测头
│   └── checkpoints/           # 模型权重
├── modules/                   # 自定义模块
├── scripts/                   # 训练/评估脚本
├── notebooks/                 # Jupyter notebooks
├── results/                   # 实验结果
│   ├── predictions/           # 预测结果
│   ├── visualizations/        # 可视化
│   └── metrics/               # 评估指标
├── logs/                      # 训练日志
├── configs/                   # 配置文件
├── docs/                      # 论文图表
└── references/                # 参考文献
```

## 快速开始

### 环境配置
```bash
conda env create -f environment.yml
conda activate yolov11
```

### 数据准备
```bash
# 下载数据集到 data/raw/
# 运行预处理脚本
python scripts/prepare_data.py
```

### 训练模型
```bash
# Baseline训练
python scripts/train.py --config configs/baseline.yaml

# 完整模型训练
python scripts/train.py --config configs/full_model.yaml
```

### 评估模型
```bash
python scripts/evaluate.py --model models/checkpoints/best.pt
```

## 实验计划

| 实验ID | 配置 | 状态 |
|--------|------|------|
| E0 | YOLOv11n baseline | ⏳ |
| E1 | +HRA-Fusion | ⏳ |
| E2 | +GD-MSE | ⏳ |
| E3 | +HD-DSAH | ⏳ |
| E4 | HRA-Fusion + GD-MSE | ⏳ |
| E5 | HRA-Fusion + HD-DSAH | ⏳ |
| E6 | GD-MSE + HD-DSAH | ⏳ |
| E7 | Full (All modules) | ⏳ |

## 更新日志

- 2026-02-07: 项目初始化
