"""
YOLOv11 Manhole Detection Modules

创新模块实现:
    - HRA-Fusion: 高分辨率自适应特征融合模块
    - GD-MSE: 梯度指导多尺度增强模块
    - HD-DSAH: 层次化解耦检测头与语义对齐机制

论文: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法
作者: [Your Name]
日期: 2026-02-08
"""

from .hra_fusion import (
    HRAFusion,
    HRAFusionNeck,
    DepthwiseSeparableConv,
    CBAM,
    BottleneckTransformer
)
from .gd_mse import (
    GDMSE,
    GDMSELite,
    GradientExtractor,
    CrossScaleAggregation,
    C3k2GD as C3k2GD_GD,
    SPPFGD as SPPFGD_GD
)
from .hd_dsah import (
    HDDSAH,
    HDDSAHMultiScale,
    HDDSAHLoss,
    ManholeStatusHierarchy,
    DecoupledHead,
    HierarchicalClassificationHead,
    SemanticAlignmentModule
)
from .model import (
    YOLOv11ManholeDetection,
    YOLOv11ManholeLoss,
    ModelFactory,
    C3k2GD,
    SPPFGD,
    DetectHD,
    GDMSE_Compatible
)

__all__ = [
    # HRA-Fusion
    "HRAFusion",
    "HRAFusionNeck",
    "DepthwiseSeparableConv",
    "CBAM",
    "BottleneckTransformer",

    # GD-MSE
    "GDMSE",
    "GDMSELite",
    "GradientExtractor",
    "CrossScaleAggregation",
    "C3k2GD_GD",
    "SPPFGD_GD",

    # HD-DSAH
    "HDDSAH",
    "HDDSAHMultiScale",
    "HDDSAHLoss",
    "ManholeStatusHierarchy",
    "DecoupledHead",
    "HierarchicalClassificationHead",
    "SemanticAlignmentModule",

    # 完整模型
    "YOLOv11ManholeDetection",
    "YOLOv11ManholeLoss",
    "ModelFactory",
    "C3k2GD",
    "SPPFGD",
    "DetectHD",
    "GDMSE_Compatible",
]

__version__ = "1.0.0"
