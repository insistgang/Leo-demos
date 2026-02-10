# Day 3 Baseline训练指南

**日期**: 2026-02-07
**状态**: ✅ 准备完成

---

## 📋 Day 3 任务清单

### 上午任务 (2小时)

| 任务 | 预计时长 | 交付物 |
|------|----------|--------|
| 检查数据集准备情况 | 30min | 数据确认 |
| 验证data.yaml配置 | 30min | 配置正确 |
| 启动baseline训练 | 30min | 训练运行 |
| 监控初始训练状态 | 30min | 确认正常 |

### 下午任务 (2小时)

| 任务 | 预计时长 | 交付物 |
|------|----------|--------|
| 配置训练日志系统 | 1h | 日志模板 |
| 设置性能监控 | 30min | 监控脚本 |
| 记录初始指标 | 30min | baseline记录 |

### 晚上任务 (1小时)

| 任务 | 预计时长 | 交付物 |
|------|----------|--------|
| 阅读技术架构报告 | 30min | 学习笔记 |
| 准备Day 4模块设计 | 30min | 设计计划 |

---

## 🚀 快速开始

### 1. 检查数据集

```bash
cd /d/jglw/yolov11-manhole-detection

# 检查数据集结构
ls data/processed/images/train | wc -l
ls data/processed/labels/train | wc -l

# 验证data.yaml
python -c "
import yaml
with open('configs/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
    print('类别数量:', data['nc'])
    print('类别列表:', data['names'])
"
```

### 2. 验证PyTorch环境

```bash
# 检查CUDA
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU数量:', torch.cuda.device_count())"

# 检查Ultralytics
python -c "import ultralytics; print('Ultralytics版本:', ultralytics.__version__)"
```

### 3. 启动训练

```bash
# 方式1: 使用train_baseline.py（推荐）
python scripts/train_baseline.py --config configs/baseline.yaml --mode train

# 方式2: 使用ultralytics命令
yolo detect train data=configs/data.yaml model=yolo11n.pt epochs=100 batch=16 imgsz=640

# 方式3: 后台运行训练
nohup yolo detect train data=configs/data.yaml model=yolo11n.pt epochs=100 batch=16 > logs/train.log 2>&1 &
```

---

## 📊 训练监控

### 实时监控命令

```bash
# 查看训练日志
tail -f runs/train/baseline_*/train.log

# 查看GPU使用情况
nvidia-smi -l 1

# 查看训练进度
ls -lh runs/train/baseline_*/weights/
```

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir runs/train

# 浏览器访问
# http://localhost:6006
```

---

## 📈 训练阶段检查

### 阶段1: 前10轮
- [ ] Loss正常下降
- [ ] mAP@0.5开始上升
- [ ] 无NaN/Inf错误
- [ ] GPU利用率 >80%

### 阶段2: 10-50轮
- [ ] Loss持续下降
- [ ] mAP@0.5稳步提升
- [ ] 验证集mAP与训练集接近

### 阶段3: 50-100轮
- [ ] 收敛趋于平稳
- [ ] 最终mAP@0.5达到预期
- [ ] 保存best.pt

---

## 🎯 预期结果

### 指标目标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| mAP@0.5 | ≥ 85% | 基础目标 |
| mAP@0.5:0.95 | ≥ 60% | COCO标准 |
| Precision | ≥ 80% | 精确度 |
| Recall | ≥ 75% | 召回率 |
| FPS | ≥ 40 | 实时性 |

### 如果未达标

| 情况 | 解决方案 |
|------|----------|
| mAP < 80% | 增加训练轮数到150-200 |
| 过拟合 | 增加数据增强，调整weight_decay |
| 欠拟合 | 减少数据增强，增大模型 |

---

## 📁 训练输出目录

```
runs/train/baseline_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt          # 最佳模型
│   ├── last.pt          # 最后模型
│   └── epoch*.pt        # 定期保存
├── results.csv
├── training_summary.json
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── PR_curve.png
├── R_curve.png
└── ...
```

---

## 🔧 故障排查

### 问题1: CUDA out of memory

**解决方案**:
```bash
# 减小batch size
# 修改 configs/baseline.yaml
batch: 8  # 原来是16

# 或使用CPU训练
device: cpu
```

### 问题2: 数据集路径错误

**解决方案**:
```bash
# 检查configs/data.yaml中的path配置
# 确保是绝对路径或相对于训练脚本的正确路径
```

### 问题3: 标签文件未找到

**解决方案**:
```bash
# 检查标签文件是否对应
ls data/processed/images/train | head -5
ls data/processed/labels/train | head -5

# 图像和标签数量应该一致
```

---

## 📝 训练记录模板

```markdown
### Day 3 训练记录

**日期**: 2026-02-07
**实验ID**: E0 - Baseline

**训练配置**:
- 模型: YOLOv11n
- 数据集: [具体数据集名称]
- 训练集: [数量]张
- 验证集: [数量]张
- Batch Size: 16
- 初始学习率: 0.001
- Epochs: 100

**训练进度**:
- 开始时间: [时间]
- 预计结束: [时间]
- 当前状态: [训练中/完成]

**结果记录**:
- 最终mAP@0.5: [数值]
- 最终mAP@0.5:0.95: [数值]
- Precision: [数值]
- Recall: [数值]
- FPS: [数值]

**备注**:
- [训练过程中的观察]
- [遇到的问题和解决]
```

---

## ⏭️ 下一步 (Day 4)

训练完成后：
1. 评估模型性能
2. 分析失败案例
3. 记录baseline指标
4. 准备HRA-Fusion模块设计

---

**Day 3 准备完成时间**: 2026-02-07
**准备好启训练**: 数据集就绪后即可开始
