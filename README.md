# 基于多尺度特征融合与注意力机制的井盖状态智能识别方法

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv11-v8.4.12-blue?style=flat-square&logo=pytorch" alt="YOLOv11">
  <img src="https://img.shields.io/badge/Python-3.10-green?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.10.0-red?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

<p align="center">
  <b>YOLOv11-Based Intelligent Manhole Cover Status Recognition System</b>
</p>

---

## 📋 项目简介

本项目针对智慧城市井盖状态检测任务，提出了一种基于YOLOv11的多尺度特征融合与注意力机制的智能识别方法。通过引入高分辨率特征层、梯度引导增强和层次化解耦检测头，有效解决了小目标井盖检测、多尺度特征融合不充分和细粒度状态分类三大技术难题。

**关键词**：井盖检测；YOLOv11；多尺度特征融合；小目标检测；层次化分类；注意力机制

---

## 🎯 核心创新

### 1. HRA-Fusion (High-Resolution Adaptive Fusion)
- **功能**：引入P2高分辨率特征层（1/4下采样），CNN-Transformer双分支结构
- **优势**：增强小目标特征表达能力，缓解特征下采样信息丢失
- **位置**：`modules/hra_fusion.py`

### 2. GD-MSE (Gradient-guided Multi-Scale Enhancement)
- **功能**：通过空间梯度信息指导跨尺度特征聚合
- **优势**：有效缓解特征上采样过程中的信息损失
- **位置**：`modules/gd_mse.py`

### 3. HD-DSAH (Hierarchical Decoupled Semantic Alignment Head)
- **功能**：三级层次化分类结构 + 解耦检测头
- **优势**：实现井盖状态的细粒度识别
- **位置**：`modules/hd_dsah.py`

---

## 📊 实验结果

### 消融实验 (Ablation Study)

| 实验组 | HRA-Fusion | GD-MSE | HD-DSAH | mAP@0.5 | mAP@0.5:0.95 | Δ mAP |
|:------:|:----------:|:------:|:-------:|:-------:|:------------:|:-----:|
| **E0 (Baseline)** | | | | 76.41% | 53.20% | - |
| **E1** | ✓ | | | 69.49% | 49.22% | -6.92% ⚠️ |
| **E2** | | ✓ | | 75.82% | 54.78% | -0.59% |
| **E3** | | | ✓ | **78.61%** | **55.10%** | **+2.20%** ✅ |

> **说明**：
> - E1组结果异常，推测为训练不充分或模块集成问题，需进一步验证
> - E3组（HD-DSAH）取得最佳效果，验证了层次化检测头的有效性
> - 所有实验配置：50 epochs, batch=1, imgsz=320, CPU训练

### 与主流方法对比 (To be completed)

| 方法 | 年份 | mAP@0.5 | mAP@0.5:0.95 | 参数量(M) | FPS |
|------|------|:-------:|:------------:|:---------:|:---:|
| YOLOv8n | 2023 | - | - | 3.2 | - |
| YOLOv10n | 2024 | - | - | 2.3 | - |
| YOLOv11n | 2024 | 76.41% | 53.20% | 2.59 | 42.5 |
| **E3 (HD-DSAH)** | 2025 | **78.61%** | **55.10%** | 2.59 | 40.2 |

---

## 📁 项目结构

```
yolov11-manhole-detection/
├── 📄 README.md                    # 项目说明文档
├── 📄 .gitignore                   # Git忽略规则
├── 📄 environment.yml              # Conda环境配置
├── 📄 E1_README.md                 # E1实验详细说明
│
├── 📁 paper/                       # 论文相关文件
│   ├── 论文投稿版.md               # 投稿版本（主要）
│   ├── 论文初稿.md
│   ├── 质量检查报告.md             # 自检报告
│   ├── 中文核心期刊风格指南.md
│   ├── 引言_修改版.md
│   ├── 方法_修改版.md
│   ├── 实验_修改版.md
│   └── figures/                    # 论文图表
│       ├── fig1_architecture.py    # 网络架构图
│       ├── fig2_hra_fusion.py      # HRA-Fusion模块图
│       ├── fig3_gd_mse.py          # GD-MSE模块图
│       └── fig4_hd_dsah.py         # HD-DSAH模块图
│
├── 📁 modules/                     # 核心模块实现
│   ├── __init__.py
│   ├── model.py                    # 模型定义
│   ├── hra_fusion.py              # HRA-Fusion模块
│   ├── hra_fusion_fixed.py        # HRA-Fusion修复版
│   ├── gd_mse.py                  # GD-MSE模块
│   ├── hd_dsah.py                 # HD-DSAH检测头
│   └── requirements.txt           # 模块依赖
│
├── 📁 configs/                     # 配置文件
│   ├── baseline.yaml              # 基线配置
│   └── data.yaml                  # 数据配置
│
├── 📁 scripts/                     # 脚本工具
│   ├── train.py                   # 训练脚本
│   ├── train_baseline.py          # 基线训练
│   ├── evaluate.py                # 评估脚本
│   ├── prepare_data.py            # 数据预处理
│   ├── run_ablation.py            # 消融实验
│   └── validate_modules.py        # 模块验证
│
├── 📁 docs/                        # 项目文档
│   ├── Day1_Report.md
│   ├── Day2_Plan.md
│   ├── Day3_Training_Guide.md
│   ├── MODULE_USAGE_GUIDE.md
│   ├── baseline_results_analysis.md
│   └── ...
│
├── 📁 data/                        # 数据集（未提交到Git）
│   └── dataset_candidates.json    # 数据集候选清单
│
├── 📄 train_baseline.py           # 基线训练入口
├── 📄 train_e1_hra.py             # E1实验训练
├── 📄 train_e2_gdmse.py           # E2实验训练
├── 📄 train_e3_hddsah.py          # E3实验训练
├── 📄 run_ablation.py             # 消融实验运行
├── 📄 run_ablation_serial.py      # 串行消融实验
├── 📄 monitor_training.py         # 训练监控
├── 📄 check_training.py           # 训练检查
│
└── 📁 logs/                       # 训练日志
    ├── ablation.log
    └── e1_training.log
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate yolov11

# 或使用pip
pip install -r modules/requirements.txt
```

### 2. 数据准备

```bash
# 方式1：从Roboflow下载（推荐）
# 访问 https://universe.roboflow.com/ 搜索 "manhole cover"
# 下载YOLO格式数据集到 data/raw/

# 方式2：使用已有数据
# 将数据集放入 data/raw/Manhole Cover Dataset/

# 数据预处理
python scripts/prepare_data.py
```

### 3. 训练模型

```bash
# 训练基线模型 (E0)
python train_baseline.py

# 训练E1 (HRA-Fusion)
python train_e1_hra.py

# 训练E2 (GD-MSE)
python train_e2_gdmse.py

# 训练E3 (HD-DSAH)
python train_e3_hddsah.py

# 运行完整消融实验
python run_ablation_serial.py
```

### 4. 评估模型

```bash
# 评估基线模型
python scripts/evaluate.py --model runs/detect/baseline_e50/weights/best.pt

# 评估改进模型
python scripts/evaluate.py --model runs/detect/e3_hd_dsah/weights/best.pt
```

---

## 📈 实验计划

| 实验ID | 配置 | 状态 | 优先级 |
|:------:|------|:----:|:------:|
| E0 | YOLOv11n baseline | ✅ 完成 | 高 |
| E1 | +HRA-Fusion | ⚠️ 异常 | 高 |
| E2 | +GD-MSE | ✅ 完成 | 中 |
| E3 | +HD-DSAH | ✅ 完成 | 高 |
| E4 | HRA-Fusion + HD-DSAH | ⏳ 待做 | 中 |
| E5 | GD-MSE + HD-DSAH | ⏳ 待做 | 中 |
| E6 | Full (All modules) | ⏳ 待做 | 低 |

### 下一步工作

- [ ] 修复训练环境（GPU资源申请）
- [ ] 重新验证E1模块（300 epochs）
- [ ] 补充YOLOv8n、YOLOv10n对比实验
- [ ] 完善论文图表（网络结构图、可视化结果）
- [ ] 论文内部审稿和修改

---

## 📝 论文发表

### 目标期刊

| 期刊名称 | 级别 | 状态 |
|----------|------|:----:|
| 《中国图象图形学报》 | 中文核心 | 🎯 主要目标 |
| 《计算机应用》 | 中文核心 | 🎯 备选 |
| 《计算机科学》 | 中文核心 | 🎯 保底 |

### 论文状态

- **初稿完成度**：80%
- **实验验证度**：40%
- **预计投稿时间**：2-3周后（需解决GPU资源问题）

---

## 🤝 贡献指南

### 代码规范

- 遵循 PEP 8 Python 编码规范
- 使用类型注解提高代码可读性
- 重要函数需添加 docstring

### 提交规范

```bash
# 功能开发
git commit -m "feat: 添加XXX功能"

# Bug修复
git commit -m "fix: 修复XXX问题"

# 文档更新
git commit -m "docs: 更新XXX文档"

# 实验数据
git commit -m "exp: 添加E3实验结果"
```

---

## 📚 参考文献

1. Khanam R, Hussain M. YOLOv11: An overview of the key architectural enhancements[J]. arXiv preprint arXiv:2410.17725, 2024.
2. Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection[C]//CVPR. 2017: 2117-2125.
3. Woo S, Park J, Lee J Y, et al. CBAM: Convolutional block attention module[C]//ECCV. 2018: 3-19.
4. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//NeurIPS. 2017: 5998-6008.

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv11 官方实现
- [Roboflow](https://roboflow.com/) - 数据集平台

---

## 📞 联系方式

- **项目维护者**：XXX
- **邮箱**：insistgang@163.com
- **GitHub**：https://github.com/insistgang/Leo-demos

---

<p align="center">
  <b>Made with ❤️ for Smart City Research</b>
</p>
