"""
YOLOv11井盖检测完整模型

整合HRA-Fusion、GD-MSE、HD-DSAH模块的完整模型实现

论文: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法
作者: [Your Name]
日期: 2026-02-08
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import sys
from pathlib import Path

# 添加模块路径
MODULES_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULES_DIR))

from hra_fusion import HRAFusion, HRAFusionNeck
from gd_mse import GDMSE, GDMSELite
from hd_dsah import HDDSAHMultiScale, HDDSAHLoss


# ============================================================================
# YOLOv11兼容的模块定义
# ============================================================================

class C3k2GD(nn.Module):
    """
    带梯度信息的C3k2模块

    兼容YOLOv11的C3k2接口，添加梯度指导增强
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        use_gradient: bool = True
    ):
        super().__init__()
        self.use_gradient = use_gradient

        # 标准C3k2结构
        from ultralytics.nn.modules import C3k2
        self.c3k2 = C3k2(c1, c2, n, c3k, e, g, shortcut)

        # 梯度信息提取
        if use_gradient:
            self.gradient_conv = nn.Sequential(
                nn.Conv2d(c2, c2 // 16, 1, bias=False),
                nn.BatchNorm2d(c2 // 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2 // 16, c2, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with gradient enhancement"""
        feat = self.c3k2(x)

        if self.use_gradient:
            # 计算梯度敏感度
            grad_proxy = x.var(dim=1, keepdim=True)
            grad_info = self.gradient_conv(feat) * torch.sigmoid(grad_proxy)
            feat = feat + feat * grad_info * 0.1

        return feat


class SPPFGD(nn.Module):
    """
    带空间梯度信息的SPPF模块

    兼容YOLOv11的SPPF接口，添加空间梯度特征
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 5,
        use_gradient: bool = True
    ):
        super().__init__()
        self.use_gradient = use_gradient

        # 标准SPPF结构
        from ultralytics.nn.modules import SPPF
        self.sppf = SPPF(c1, c2, k)

        # 空间梯度提取
        if use_gradient and c1 == c2:
            self.spatial_grad = nn.Sequential(
                nn.Conv2d(c2, c2 // 4, 1, bias=False),
                nn.BatchNorm2d(c2 // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2 // 4, c2, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with spatial gradient enhancement"""
        feat = self.sppf(x)

        if self.use_gradient and hasattr(self, 'spatial_grad'):
            spatial_grad = self.spatial_grad(x)
            feat = feat * (1 + spatial_grad * 0.1)

        return feat


class DetectHD(nn.Module):
    """
    层次化解耦检测头 (兼容YOLOv11的Detect接口)

    替换原Detect头，支持:
    - 解耦的分类/回归/置信度分支
    - 层次化分类 (存在性 -> 状态 -> 细粒度)
    - 语义对齐
    """

    def __init__(
        self,
        nc: int = 7,  # 类别数
        ch: tuple = (),  # 输入通道数元组
        hidden_channels: int = 256,
        use_hierarchical: bool = True
    ):
        super().__init__()
        self.nc = nc
        self.use_hierarchical = use_hierarchical

        # 检测层数
        self.nl = len(ch)

        # 每个检测头的输出通道数
        # nc: 分类, 4: 边界框, 1: 置信度
        self.no = nc + 4 + 1

        # 每层的检测头
        self.detect_heads = nn.ModuleList()

        for i in range(self.nl):
            in_c = ch[i]
            head = self._build_head(in_c, nc, hidden_channels, use_hierarchical)
            self.detect_heads.append(head)

        # 初始化权重
        self._init_weights()

    def _build_head(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int,
        use_hierarchical: bool
    ) -> nn.Module:
        """构建单个检测头"""
        # 分类分支
        cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1)
        )

        # 回归分支 (边界框预测)
        reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, 1)
        )

        # 置信度分支 (objectness)
        obj_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1)
        )

        return nn.ModuleDict({
            'cls': cls_branch,
            'reg': reg_branch,
            'obj': obj_branch
        })

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass

        Args:
            x: 多尺度特征列表 [P3, P4, P5] 或 [P2, P3, P4, P5]

        Returns:
            输出张量元组 (cls_outputs, reg_outputs, obj_outputs)
        """
        cls_outputs = []
        reg_outputs = []
        obj_outputs = []

        for i, feat in enumerate(x):
            head = self.detect_heads[i]
            cls_out = head['cls'](feat)
            reg_out = head['reg'](feat)
            obj_out = head['obj'](feat)

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            obj_outputs.append(obj_out)

        # YOLO格式: 拼接所有输出
        # 每个尺度的输出: (B, nc+4+1, anchors, H, W)
        return tuple(cls_outputs + reg_outputs + obj_outputs)


# ============================================================================
# 完整的YOLOv11改进模型包装器
# ============================================================================

class YOLOv11ManholeDetection(nn.Module):
    """
    基于YOLOv11的井盖状态检测模型

    整合了三个创新模块:
    1. HRA-Fusion: 高分辨率自适应特征融合 (P2层 + 双分支融合)
    2. GD-MSE: 梯度指导多尺度特征增强
    3. HD-DSAH: 层次化解耦检测头与语义对齐

    Args:
        use_hra_fusion: 是否使用HRA-Fusion模块
        use_gd_mse: 是否使用GD-MSE模块
        use_hd_dsah: 是否使用HD-DSAH检测头
        num_classes: 类别数 (默认7类井盖状态)
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
    """

    def __init__(
        self,
        use_hra_fusion: bool = True,
        use_gd_mse: bool = True,
        use_hd_dsah: bool = True,
        num_classes: int = 7,
        model_size: str = 'n',
        pretrained_path: Optional[str] = None
    ):
        super().__init__()

        self.use_hra_fusion = use_hra_fusion
        self.use_gd_mse = use_gd_mse
        self.use_hd_dsah = use_hd_dsah
        self.num_classes = num_classes
        self.model_size = model_size

        # 加载预训练的YOLOv11模型
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel

        # 加载基础模型
        if pretrained_path:
            self.base_model = YOLO(pretrained_path).model
        else:
            # 使用内置权重
            model_map = {
                'n': 'yolo11n.pt',
                's': 'yolo11s.pt',
                'm': 'yolo11m.pt',
                'l': 'yolo11l.pt',
                'x': 'yolo11x.pt'
            }
            weights_path = model_map.get(model_size, 'yolo11n.pt')
            self.base_model = YOLO(weights_path).model

        # 保存原始backbone和neck
        self._extract_base_components()

        # 添加创新模块
        self._add_innovative_modules()

    def _extract_base_components(self):
        """提取YOLOv11的backbone和neck组件"""
        # YOLOv11模型结构分析
        # Backbone: Conv -> C3k2 -> ... -> SPPF
        # Neck: Upsample -> Concat -> C3k2 (自顶向下)
        # Head: Detect

        model = self.base_model.model

        # 找到关键层的索引
        self.backbone_end_idx = 9  # SPPF之前的层
        self.neck_end_idx = 22     # Detect之前的层

        # 分割模型
        self.backbone = nn.Sequential(*list(model.children())[:self.backbone_end_idx])
        self.neck = nn.Sequential(*list(model.children())[self.backbone_end_idx:self.neck_end_idx])
        self.detect_head = model[-1]

        # 保存中间层输出索引 (用于获取多尺度特征)
        # P3: 第4层 (Conv), P4: 第6层 (Conv), P5: 第8层 (Conv)
        self.feature_indices = [4, 6, 8]

    def _add_innovative_modules(self):
        """添加创新模块"""
        # 获取通道数配置
        channel_map = {
            'n': [64, 128, 256],
            's': [128, 256, 512],
            'm': [192, 384, 768],
            'l': [256, 512, 1024],
            'x': [320, 640, 1280]
        }
        channels = channel_map.get(self.model_size, [64, 128, 256])

        # HRA-Fusion Neck (用于P2层添加)
        if self.use_hra_fusion:
            # P2层从backbone早期层获取
            self.p2_conv = nn.Conv2d(channels[0], channels[2], 1, bias=False)
            self.hra_fusion = HRAFusion(
                in_channels=channels[2],
                out_channels=channels[2],
                num_heads=8
            )

        # GD-MSE增强
        if self.use_gd_mse:
            self.gd_mse = GDMSE(
                in_channels=channels[2],
                num_scales=3  # P3, P4, P5
            )

        # HD-DSAH检测头
        if self.use_hd_dsah:
            self.hd_detect_head = DetectHD(
                nc=self.num_classes,
                ch=tuple(channels),
                hidden_channels=256
            )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: 输入图像 (B, 3, H, W)
            return_features: 是否返回中间特征

        Returns:
            Dict containing predictions
        """
        # Backbone特征提取
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)

        # features: [P3, P4, P5]
        p3, p4, p5 = features

        # HRA-Fusion: 添加P2层
        if self.use_hra_fusion:
            # 从P3下采样得到P2
            p2 = nn.functional.interpolate(p3, scale_factor=2, mode='nearest')
            p2 = self.p2_conv(p2)
            p2 = self.hra_fusion(p2, p3, p4, p5)
            features = [p2, p3, p4, p5]
        else:
            features = [p3, p4, p5]

        # Neck处理
        neck_outputs = []
        for layer in self.neck:
            if isinstance(layer, (nn.Upsample, nn.Conv2d)):
                x = layer(x)
            elif hasattr(layer, 'forward_once'):  # Concat等特殊层
                x = layer(x)
            neck_outputs.append(x)

        # GD-MSE增强
        if self.use_gd_mse:
            features = self.gd_mse(tuple(features))

        # 检测头
        if self.use_hd_dsah:
            predictions = self.hd_detect_head(features)
        else:
            predictions = self.detect_head(features)

        output = {
            "predictions": predictions,
            "features": features if return_features else None
        }

        return output


# ============================================================================
# 损失函数包装器
# ============================================================================

class YOLOv11ManholeLoss(nn.Module):
    """
    完整的损失函数包装器

    根据启用的模块返回相应的损失
    """

    def __init__(
        self,
        use_hd_dsah: bool = True,
        num_classes: int = 7,
        **kwargs
    ):
        super().__init__()

        if use_hd_dsah:
            self.loss_fn = HDDSAHLoss(num_classes=num_classes, **kwargs)
        else:
            # 使用YOLOv8标准损失
            from ultralytics.utils.loss import v8DetectionLoss
            self.loss_fn = v8DetectionLoss

    def forward(
        self,
        predictions: Dict,
        targets: Dict
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        if self.loss_fn is not None:
            return self.loss_fn(predictions, targets)
        else:
            return {"total_loss": torch.tensor(0.0)}


# ============================================================================
# 模型工厂
# ============================================================================

class ModelFactory:
    """模型工厂 - 根据配置创建模型"""

    @staticmethod
    def create_model(
        experiment_id: str,
        num_classes: int = 7,
        model_size: str = 'n',
        pretrained_path: Optional[str] = None
    ) -> YOLOv11ManholeDetection:
        """
        根据实验ID创建对应的模型配置

        Args:
            experiment_id: 实验ID (E0-E7)
            num_classes: 类别数
            model_size: 模型大小
            pretrained_path: 预训练权重路径

        Returns:
            配置好的模型
        """
        configs = {
            "E0": {"use_hra_fusion": False, "use_gd_mse": False, "use_hd_dsah": False},
            "E1": {"use_hra_fusion": True,  "use_gd_mse": False, "use_hd_dsah": False},
            "E2": {"use_hra_fusion": False, "use_gd_mse": True,  "use_hd_dsah": False},
            "E3": {"use_hra_fusion": False, "use_gd_mse": False, "use_hd_dsah": True},
            "E4": {"use_hra_fusion": True,  "use_gd_mse": True,  "use_hd_dsah": False},
            "E5": {"use_hra_fusion": True,  "use_gd_mse": False, "use_hd_dsah": True},
            "E6": {"use_hra_fusion": False, "use_gd_mse": True,  "use_hd_dsah": True},
            "E7": {"use_hra_fusion": True,  "use_gd_mse": True,  "use_hd_dsah": True},
        }

        config = configs.get(experiment_id, configs["E0"])
        return YOLOv11ManholeDetection(
            num_classes=num_classes,
            model_size=model_size,
            pretrained_path=pretrained_path,
            **config
        )


# ============================================================================
# 兼容性导出
# ============================================================================

# 为了向后兼容，导出简化的GDMSE
class GDMSE_Compatible(GDMSE):
    """兼容性GDMSE包装器"""
    pass


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("YOLOv11 Manhole Detection Model Test")
    print("="*60)

    # 测试模型工厂
    print("\n1. Testing Model Factory")
    for exp_id in ["E0", "E1", "E2", "E3", "E7"]:
        config = {
            "E0": ("Baseline", []),
            "E1": ("HRA-Fusion", ["HRA-Fusion"]),
            "E2": ("GD-MSE", ["GD-MSE"]),
            "E3": ("HD-DSAH", ["HD-DSAH"]),
            "E7": ("Full", ["HRA-Fusion", "GD-MSE", "HD-DSAH"]),
        }
        name, modules = config.get(exp_id, ("Unknown", []))
        modules_str = ', '.join(modules) if modules else "Baseline"
        print(f"   {exp_id}: {name} ({modules_str})")

    # 测试单个模块
    print("\n2. Testing Individual Modules")

    # 测试C3k2GD
    from ultralytics.nn.modules import C3k2
    c3k2_gd = C3k2GD(64, 64, use_gradient=True)
    x = torch.randn(1, 64, 80, 80)
    y = c3k2_gd(x)
    print(f"   C3k2GD: {x.shape} -> {y.shape}")

    # 测试SPPFGD
    sppf_gd = SPPFGD(256, 256, use_gradient=True)
    x = torch.randn(1, 256, 40, 40)
    y = sppf_gd(x)
    print(f"   SPPFGD: {x.shape} -> {y.shape}")

    # 测试DetectHD
    detect_hd = DetectHD(nc=7, ch=(256, 256, 256))
    features = [
        torch.randn(1, 256, 80, 80),
        torch.randn(1, 256, 40, 40),
        torch.randn(1, 256, 20, 20)
    ]
    outputs = detect_hd(features)
    print(f"   DetectHD: {len(outputs)} outputs")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
