# 论文引用文献整理

> **论文题目**: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法
> **目标期刊**: 中国图象图形学报 / 计算机应用
> **整理日期**: 2026-02-08

---

## 一、按引用位置分类的参考文献

### 1. 引言部分引用文献

#### 1.1 智慧城市与基础设施管理
- [ ] 智慧城市建设相关政策文献
- [ ] 井盖安全事故统计数据来源

#### 1.2 YOLO系列发展
- [1] Redmon J, et al. You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016.
- [2] Redmon J, Farhadi A. YOLO9000: Better, Faster, Stronger. CVPR 2017.
- [3] Redmon J, Farhadi A. YOLOv3: An Incremental Improvement. arXiv 2018.
- [4] Bochkovskiy A, et al. YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv 2020.
- [5] Ge Z, et al. YOLOX: Exceeding YOLO Series in 2021. CVPR 2022.
- [6] Ultralytics. YOLOv8: An Open Source Project for Object Detection. GitHub 2023.
- [7] Wang C Y, Liao H Y M. YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information. ECCV 2024.
- [8] Wang A, et al. YOLOv10: Real-Time End-to-End Object Detection. arXiv 2024.
- [9] Ultralytics. YOLOv11: An Overview of the Key Architectural Analysis. arXiv:2410.17725, 2024.

#### 1.3 井盖检测相关研究
- [10] 郑婉茹, 等. 基于改进YOLOv8算法的井盖检测研究. 计算机科学与应用, 2025.
- [11] MGB-YOLO. Real-time Detection of Road Manhole Covers Using MGB-YOLO. Nature Scientific Reports, 2023.
- [12] EEFA-YOLO. Road Manhole Cover Defect Detection via Multi-Scale Edge Feature Aggregation. Nature Scientific Reports, 2025.

### 2. 相关工作部分引用文献

#### 2.1 经典检测器
- [13] Ren S, et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS 2015.
- [14] Liu W, et al. SSD: Single Shot MultiBox Detector. ECCV 2016.
- [15] Lin T Y, et al. Focal Loss for Dense Object Detection. ICCV 2017.

#### 2.2 小目标检测技术
- [16] Lin T Y, et al. Feature Pyramid Networks for Object Detection. CVPR 2017.
- [17] Tan M, et al. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020.
- [18] Sun H, et al. Uncertainty-Aware Gradient Stabilization for Small Object Detection. ICCV 2025.
- [19] Xie J, et al. Complementary Feature Pyramid Network for Object Detection. ACM TOMM 2023.

#### 2.3 注意力机制
- [20] Vaswani A, et al. Attention Is All You Need. NeurIPS 2017.
- [21] Woo S, et al. CBAM: Convolutional Block Attention Module. ECCV 2018.
- [22] Hu J, et al. Squeeze-and-Excitation Networks. CVPR 2018.
- [23] Hou Q, et al. Coordinate Attention for Efficient Mobile Network Design. CVPR 2021.

#### 2.4 Transformer在检测中的应用
- [24] Carion N, et al. End-to-End Object Detection with Transformers. ECCV 2020.
- [25] Liu Z, et al. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. ICCV 2021.
- [26] Lv W, et al. DETRs Beat YOLOs on Real-time Object Detection. CVPR 2023.

### 3. 方法部分引用文献

#### 3.1 YOLOv11架构
- [9] Ultralytics. YOLOv11: An Overview of the Key Architectural Analysis. arXiv:2410.17725, 2024.

#### 3.2 Gold-YOLO (GD机制参考)
- [27] Tian Y, et al. Gold-YOLO: Efficient Object Detection via Gather-and-Distribute Mechanism. NeurIPS 2024.

#### 3.3 层次化分类
- [28] Task-Aligned Learning for Object Detection. 相关文献

#### 3.4 解耦检测头
- [5] Ge Z, et al. YOLOX: Exceeding YOLO Series in 2021. CVPR 2022. (YOLOX解耦头)

### 4. 实验部分引用文献

#### 4.1 数据集
- [29] Lin T Y, et al. Microsoft COCO: Common Objects in Context. ECCV 2014.

#### 4.2 评估指标
- 标准引用COCO评估指标文献

---

## 二、按重要性分类的参考文献

### A类：必须引用（核心文献）

#### YOLO系列
1. [ultralytics2024yolov11] YOLOv11技术报告
2. [ge2022yolox] YOLOX（解耦头参考）
3. [redmon2016yolo] YOLOv1（奠基性工作）

#### 小目标检测
4. [lin2017fpn] FPN（特征金字塔基础）
5. [sun2025ugs] ICCV 2025小目标检测最新进展

#### 注意力机制
6. [woo2018cbam] CBAM（本文使用）

#### 井盖检测
7. [zheng2025yolov8] 郑婉茹2025（直接对比）

### B类：重要引用（支撑性文献）

1. [wang2024yolov9] YOLOv9 (PGI机制参考)
2. [tian2024goldyolo] Gold-YOLO (GD机制参考)
3. [li2025mfayolo] MFA-YOLO (多特征聚合参考)

### C类：可选引用（扩展性文献）

1. 其他YOLO变体（YOLOv5, YOLOv7, YOLOv10）
2. 其他注意力机制（SE-Net, Coordinate Attention）
3. Transformer相关（DETR, Swin, RT-DETR）

---

## 三、参考文献GB/T 7714-2015格式模板

### 3.1 期刊文章格式
```
[序号] 作者. 文章标题[J]. 期刊名称, 年份, 卷(期): 起止页码.
```

示例：
```
[6] 郑婉茹, 陈天顺, 熊天赐, 等. 基于改进YOLOv8算法的井盖检测研究[J]. 计算机科学与应用, 2025, 15(12): 77-90.
```

### 3.2 会议论文格式
```
[序号] 作者. 论文标题[C]//会议名称. 出版地: 出版社, 年份: 起止页码.
```

示例：
```
[1] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Las Vegas: IEEE, 2016: 779-788.
```

### 3.3 预印本格式
```
[序号] 作者. 论文标题[EB/OL]. (发布日期)[访问日期]. 获取和访问路径.
```

示例：
```
[9] Ultralytics. YOLOv11: An Overview of the Key Architectural Analysis[EB/OL]. (2024-10-23)[2026-02-08]. https://arxiv.org/abs/2410.17725.
```

---

## 四、引用文献统计

### 4.1 分类统计
| 类别 | 数量 | 说明 |
|------|------|------|
| YOLO系列 | 9篇 | YOLOv1-v11完整演进 |
| 经典检测器 | 3篇 | Faster R-CNN, SSD, Focal Loss |
| 小目标检测 | 4篇 | FPN, CFPN, UGS等 |
| 注意力机制 | 4篇 | Transformer, CBAM, SE-Net等 |
| 井盖检测 | 3篇 | 直接相关工作 |
| Transformer | 3篇 | DETR, Swin, RT-DETR |
| **总计** | **26+** | 核心引用 |

### 4.2 年份分布
| 年份 | 数量 | 占比 |
|------|------|------|
| 2025 | 3篇 | 最前沿 |
| 2024 | 5篇 | 最新 |
| 2022-2023 | 8篇 | 近期 |
| 2017-2021 | 10篇 | 经典 |

---

## 五、引用策略建议

### 5.1 引言部分引用策略
- 第一段：智慧城市背景 → 引用政策和统计资料
- 第二段：YOLO发展 → 引用YOLOv1-v11关键论文
- 第三段：井盖检测现状 → 引用郑婉茹2025等3篇

### 5.2 相关工作引用策略
- 目标检测发展：按时间线引用（Faster R-CNN → SSD → YOLO → Transformer）
- 小目标检测技术：重点引用FPN、CFPN、UGS
- 井盖检测：对比3-4篇相关工作，指出空白

### 5.3 方法部分引用策略
- YOLOv11：必须引用最新arXiv论文
- 注意力机制：引用CBAM（本文使用）
- 解耦头：引用YOLOX

### 5.4 实验部分引用策略
- 对比方法：必须引用原始论文
- 评估指标：引用COCO论文

---

## 六、文献质量控制

### 6.1 来源可靠性
- [ ] 顶级会议（CVPR, ICCV, ECCV, NeurIPS）：优先引用
- [ ] 顶级期刊（TPAMI, IJCV）：优先引用
- [ ] 权威arXiv：引用需谨慎，优先选择已被接受的版本

### 6.2 时效性
- [ ] 2024-2025年文献：占比>30%
- [ ] 2020-2023年文献：占比>50%
- [ ] 经典文献（2017年前）：选择性引用奠基性工作

### 6.3 引用平衡
- [ ] 自引（团队前期工作）：<10%
- [ ] 他引：>90%
- [ ] 中文文献：适当引用（国内期刊要求）

---

## 七、投稿不同期刊的引用策略

### 7.1 《中国图象图形学报》
- 适当增加中文核心期刊引用
- GB/T 7714-2015格式
- 建议中文文献占比20-30%

### 7.2 《计算机应用》
- 强调应用价值和实用性
- 引用同类应用研究
- 增加工程实现相关引用

### 7.3 英文SCI期刊
- 增加顶级会议/期刊引用
- IEEE或Springer格式
- 以英文文献为主

---

## 八、待补充文献清单

- [ ] 智慧城市建设相关政策文献
- [ ] 井盖安全事故统计数据来源
- [ ] 更多2024-2025年小目标检测前沿研究
- [ ] 层次化分类理论基础
- [ ] 边缘设备部署相关研究

---

**整理完成时间**: 2026-02-08
**状态**: 引用文献框架完成，待实验后补充
**下一步**: 根据实验结果调整对比方法引用
