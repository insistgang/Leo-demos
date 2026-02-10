# YOLOv11n Baseline训练结果分析报告

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | YOLOv11n |
| Epochs | 50 |
| Batch Size | 1 |
| Image Size | 320x320 |
| 设备 | CPU |
| 训练集 | 3,243 张图像 |
| 验证集 | 174 张图像 |
| 类别数 | 4 (Broken, Good, Lose, Uncovered) |

## 训练结果

### 总体指标

| 指标 | 最佳值 | Epoch |
|------|--------|-------|
| **mAP@0.5** | **76.41%** | Epoch 39 |
| **mAP@0.5:0.95** | **53.20%** | Epoch 39 |
| **Precision** | **80.33%** | Epoch 50 |
| **Recall** | **70.75%** | Epoch 49 |

### 训练过程

- **总训练时间**: 约5,421秒 (90.4分钟)
- **平均每Epoch**: 约108秒
- **最佳权重**: `runs/detect/runs/train/baseline_e50/weights/best.pt` (5.17 MB)

### Loss变化趋势

| Loss | 初始值 | 最终值 | 下降幅度 |
|------|--------|--------|----------|
| train/box_loss | 1.3985 | 0.8826 | 36.9% |
| train/cls_loss | 2.7893 | 0.6080 | 78.2% |
| train/dfl_loss | 1.3766 | 1.0234 | 25.6% |

## 消融实验设计

基于Baseline结果，将进行以下消融实验：

| 实验ID | 配置 | 说明 |
|--------|------|------|
| E0 | YOLOv11n | Baseline (已完成) |
| E1 | YOLOv11n + HRA-Fusion | 仅添加高分辨率融合模块 |
| E2 | YOLOv11n + GD-MSE | 仅添加梯度引导增强模块 |
| E3 | YOLOv11n + HD-DSAH | 仅替换检测头 |
| E4 | HRA-Fusion + GD-MSE | 双模块组合 |
| E5 | HRA-Fusion + HD-DSAH | 双模块组合 |
| E6 | GD-MSE + HD-DSAH | 双模块组合 |
| E7 | Full Model | 全部三个模块 |

## 下一步工作

1. 实现并集成HRA-Fusion模块
2. 实现并集成GD-MSE模块
3. 实现并集成HD-DSAH检测头
4. 运行消融实验E1-E7
5. 对比分析各模块贡献度
6. 撰写完整论文

---
*报告生成时间: 2026-02-08*
