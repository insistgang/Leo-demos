# 实验启动指南 - 2026-02-08

> **目标**: 下载数据集并启动Baseline训练

---

## 🚀 快速启动 (3步)

### 步骤1: 查看数据集并下载

```bash
# 进入项目目录
cd D:\jglw\yolov11-manhole-detection

# 查看可用数据集
python scripts/download_roboflow.py --list
```

**推荐数据集: SideSeeing Manhole Dataset**

| 属性 | 值 |
|------|---|
| **图像数** | 1,427张 |
| **类别** | 4类 (Broken, Loose, Uncovered, Good) |
| **格式** | YOLO |
| **链接** | https://universe.roboflow.com/sideseeing/manhole-cover-dataset-yolo-62sri |

**手动下载步骤**:
1. 访问: https://universe.roboflow.com/sideseeing/manhole-cover-dataset-yolo-62sri
2. 点击 "Download" 按钮
3. 选择 "YOLOv8" 格式
4. 下载并解压到 `data/raw/sideseeing/`

### 步骤2: 配置环境

```bash
# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate yolov11

# 验证PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

# 验证Ultralytics
python -c "import ultralytics; print('Ultralytics版本:', ultralytics.__version__)"
```

### 步骤3: 启动训练

```bash
# 数据预处理 (如果需要)
python scripts/prepare_data.py --raw_dir data/raw/sideseeing --output_dir data/processed

# 检查数据质量
python scripts/check_dataset.py --check_all --base_dir data/processed

# 启动baseline训练
python scripts/train_baseline.py --config configs/baseline.yaml --mode train
```

---

## 📊 预期训练配置

| 参数 | 值 |
|------|---|
| 模型 | YOLOv11n |
| Epochs | 100 |
| Batch Size | 16 |
| 图像尺寸 | 640×640 |
| 优化器 | AdamW |
| 学习率 | 0.001 |
| 设备 | GPU (或CPU) |

---

## ⏱️ 预计时间

| 阶段 | 时间 (GPU) | 时间 (CPU) |
|------|------------|------------|
| 数据下载 | 10分钟 | 10分钟 |
| 环境配置 | 5分钟 | 5分钟 |
| Baseline训练 | 2-4小时 | 8-12小时 |

---

## 📁 训练输出

训练完成后，结果将保存在:
```
runs/train/baseline_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt          # 最佳模型
│   └── last.pt          # 最后模型
├── results.csv          # 训练结果
├── confusion_matrix.png # 混淆矩阵
└── training_curves.png  # 训练曲线
```

---

## ✅ 检查清单

- [ ] 数据集下载完成
- [ ] conda环境创建成功
- [ ] PyTorch和CUDA验证通过
- [ ] 数据质量检查通过
- [ ] Baseline训练启动成功

---

## 🔧 故障排查

### 问题1: conda环境创建失败
```bash
# 手动创建
conda create -n yolov11 python=3.10 -y
conda activate yolov11
pip install ultralytics torch torchvision opencv-python
```

### 问题2: CUDA不可用
```bash
# 检查GPU
nvidia-smi

# 使用CPU训练 (修改configs/baseline.yaml)
device: cpu
```

### 问题3: 数据集格式问题
```bash
# 检查数据集结构
ls data/processed/images/train
ls data/processed/labels/train

# 确保图像和标签数量一致
```

---

**准备好开始实验了吗？**
