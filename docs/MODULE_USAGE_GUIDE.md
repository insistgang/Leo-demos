# YOLOv11创新模块使用指南

## 项目结构

```
yolov11-manhole-detection/
├── modules/                    # 创新模块实现
│   ├── __init__.py            # 模块导出
│   ├── hra_fusion.py          # HRA-Fusion模块
│   ├── gd_mse.py              # GD-MSE模块
│   ├── hd_dsah.py             # HD-DSAH模块
│   └── model.py               # 完整模型定义
├── scripts/                   # 脚本工具
│   ├── quick_test.py          # 快速验证脚本
│   ├── validate_modules.py    # 完整验证脚本
│   └── integrate_modules.py   # 集成脚本
├── docs/                      # 文档
│   └── MODULE_VALIDATION_REPORT.md
└── data/                      # 数据集目录
```

---

## 快速开始

### 1. 验证模块安装

```bash
# 快速验证所有模块
python scripts/quick_test.py

# 完整验证（包含性能测试）
python scripts/validate_modules.py
```

### 2. 基本使用

```python
from modules import (
    HRAFusion, HRAFusionNeck,
    GDMSE, GDMSELite,
    HDDSAH, HDDSAHMultiScale,
    YOLOv11ManholeDetection, ModelFactory
)
import torch

# 创建模型
model = ModelFactory.create_model("E7", num_classes=7)

# 前向传播
x = torch.randn(1, 3, 640, 640)
predictions = model(x)
```

---

## 模块详解

### HRA-Fusion (高分辨率自适应特征融合)

**用途**: 提升小目标检测能力

```python
from modules import HRAFusion, HRAFusionNeck

# 核心模块
hra_fusion = HRAFusion(
    in_channels=256,
    out_channels=256,
    num_heads=8
)

# 完整Neck
neck = HRAFusionNeck(
    in_channels=(64, 128, 256, 512),
    out_channels=256
)

# 前向传播
p2, p3, p4, p5 = neck(x_p2, x_p3, x_p4, x_p5)
```

**特点**:
- 新增P2层（1/4下采样）
- CNN+Transformer双分支
- CBAM注意力机制
- 自适应权重融合

---

### GD-MSE (梯度指导多尺度增强)

**用途**: 增强多尺度特征表示

```python
from modules import GDMSE, GDMSELite

# 完整版
gd_mse = GDMSE(in_channels=256, num_scales=4)

# 轻量版（适合边缘部署）
gd_mse_lite = GDMSELite(in_channels=256, num_scales=4)

# 前向传播
features = (feat_p2, feat_p3, feat_p4, feat_p5)
enhanced = gd_mse(features)
```

**特点**:
- 梯度信息提取
- 跨尺度聚合
- 轻量级版本支持

---

### HD-DSAH (层次化解耦检测头)

**用途**: 井盖状态精细分类

```python
from modules import HDDSAH, HDDSAHMultiScale, HDDSAHLoss

# 单尺度检测头
head = HDDSAH(in_channels=256, num_classes=7)
outputs = head(x, return_hierarchical=True)

# 多尺度检测头
multi_head = HDDSAHMultiScale(in_channels=256, num_classes=7, num_scales=4)
outputs = multi_head(features)

# 损失函数
loss_fn = HDDSAHLoss(num_classes=7, use_hierarchical=True)
losses = loss_fn(predictions, targets)
```

**7类状态**:
- 0: intact (完整)
- 1: minor_damaged (轻度破损)
- 2: medium_damaged (中度破损)
- 3: severe_damaged (重度破损)
- 4: missing (缺失)
- 5: displaced (移位)
- 6: occluded (遮挡)

**特点**:
- 层次化三分类
- 解耦检测头
- 语义对齐机制

---

## 实验配置

使用ModelFactory创建不同配置的模型：

```python
from modules import ModelFactory

# E0: 基线
model = ModelFactory.create_model("E0")

# E1: HRA-Fusion
model = ModelFactory.create_model("E1")

# E2: GD-MSE
model = ModelFactory.create_model("E2")

# E3: HD-DSAH
model = ModelFactory.create_model("E3")

# E7: 完整模型
model = ModelFactory.create_model("E7")
```

---

## YOLOv11兼容模块

这些模块可以直接替换YOLOv11中的对应组件：

```python
from modules.model import C3k2GD, SPPFGD, DetectHD

# 替换C3k2
c3k2_gd = C3k2GD(c1=64, c2=64, use_gradient=True)

# 替换SPPF
sppf_gd = SPPFGD(c1=256, c2=256, use_gradient=True)

# 替换Detect
detect_hd = DetectHD(nc=7, ch=(256, 256, 256))
```

---

## 训练示例

### 使用ultralytics训练

```python
from ultralytics import YOLO
from modules.model import C3k2GD, SPPFGD, DetectHD

# 加载预训练模型
model = YOLO('yolo11n.pt')

# 修改模型配置以集成创新模块
# (具体实现参考 scripts/integrate_modules.py)

# 训练
results = model.train(
    data='data/manhole/data.yaml',
    epochs=100,
    batch=16,
    imgsz=640
)
```

### 使用集成脚本

```bash
# 运行完整模型实验
python scripts/integrate_modules.py --experiment E7 --epochs 100

# 运行所有实验
python scripts/integrate_modules.py --all
```

---

## 性能指标

| 模块 | 参数量 | 前向时间(CPU) | 特点 |
|------|--------|---------------|------|
| HRA-Fusion | 1.38M | ~27ms | 小目标增强 |
| GD-MSE | 4.37M | - | 多尺度增强 |
| GD-MSE Lite | 1.05M | - | 轻量版本 |
| HD-DSAH | 3.68M | ~18ms | 层次化检测 |

---

## 注意事项

1. **内存使用**: Transformer模块（HRA-Fusion）在大型输入上可能消耗较多内存
2. **轻量版本**: 对于边缘部署，建议使用GDMSELite
3. **类别配置**: 确保num_classes与数据集一致
4. **输入尺寸**: 建议使用640x640或更大输入尺寸以获得最佳性能

---

## 故障排除

### 导入错误
```bash
# 确保项目路径在PYTHONPATH中
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA内存不足
```python
# 使用更小的批次大小或减少输入尺寸
model.train(batch=8, imgsz=320)
```

### 模块验证失败
```bash
# 运行验证脚本检查
python scripts/quick_test.py
```

---

## 参考文献

论文: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法

相关技术:
- CBAM: Convolutional Block Attention Module
- YOLOv9: PGI (Programmable Gradient Information)
- ICCV 2025: Uncertainty-Aware Gradient Stabilization

---

**最后更新**: 2026-02-08
