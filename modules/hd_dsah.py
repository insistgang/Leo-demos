"""
HD-DSAH Module: Hierarchical Decoupled Detection Head with Semantic Alignment

层次化解耦检测头与语义对齐机制

论文: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法
作者: [Your Name]
日期: 2026-02-08

设计思路:
    1. 层次化三分类策略 (存在性 → 状态 → 细粒度分级)
    2. 解耦检测头 (分类与回归分支独立)
    3. 语义对齐损失 (KL散度 + 边界一致性)

7类状态定义:
    0: 完整 (intact)
    1: 轻度破损 (minor_damaged)
    2: 中度破损 (medium_damaged)
    3: 重度破损 (severe_damaged)
    4: 缺失 (missing)
    5: 移位 (displaced)
    6: 遮挡 (occluded)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


# ============================================================================
# 层次化分类定义
# ============================================================================

class ManholeStatusHierarchy:
    """
    井盖状态层次化分类定义

    Level 1: 存在性判断 (有/无)
    Level 2: 状态分类 (完好/破损/缺失)
    Level 3: 细粒度分级 (7类)
    """

    # 类别到层级的映射
    CLASS_TO_LEVEL = {
        0: (1, 0, 0),  # 完整 → 存在 → 完好 → 完整
        1: (1, 1, 0),  # 轻度破损 → 存在 → 破损 → 轻度
        2: (1, 1, 1),  # 中度破损 → 存在 → 破损 → 中度
        3: (1, 1, 2),  # 重度破损 → 存在 → 破损 → 重度
        4: (0, 2, 0),  # 缺失 → 不存在 → 缺失 → 缺失
        5: (1, 0, 1),  # 移位 → 存在 → 完好 → 移位
        6: (1, 0, 2),  # 遮挡 → 存在 → 完好 → 遮挡
    }

    # 层级到类别的映射
    LEVEL_TO_CLASS = {
        (1, 0, 0): 0,  # 完整
        (1, 1, 0): 1,  # 轻度破损
        (1, 1, 1): 2,  # 中度破损
        (1, 1, 2): 3,  # 重度破损
        (0, 2, 0): 4,  # 缺失
        (1, 0, 1): 5,  # 移位
        (1, 0, 2): 6,  # 遮挡
    }

    # 类别名称
    CLASS_NAMES = [
        "intact",          # 0
        "minor_damaged",   # 1
        "medium_damaged",  # 2
        "severe_damaged",  # 3
        "missing",         # 4
        "displaced",       # 5
        "occluded"         # 6
    ]


# ============================================================================
# 解耦检测头组件
# ============================================================================

class DecoupledHead(nn.Module):
    """
    解耦检测头

    分类分支与回归分支独立处理
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 7,
        hidden_channels: int = 256
    ):
        super().__init__()

        # 分类分支
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1)
        )

        # 回归分支 (边界框预测)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, 1)  # x, y, w, h
        )

        # 置信度分支 (objectness)
        self.obj_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            cls_pred: 分类预测 (B, num_classes, H, W)
            reg_pred: 回归预测 (B, 4, H, W)
            obj_pred: 置信度预测 (B, 1, H, W)
        """
        cls_pred = self.cls_branch(x)
        reg_pred = self.reg_branch(x)
        obj_pred = self.obj_branch(x)

        return cls_pred, reg_pred, obj_pred


# ============================================================================
# 层次化检测头
# ============================================================================

class HierarchicalClassificationHead(nn.Module):
    """
    层次化分类头

    三层级分类:
    - Level 1: 存在性 (2类: 有/无)
    - Level 2: 状态 (3类: 完好/破损/缺失)
    - Level 3: 细粒度 (7类)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256
    ):
        super().__init__()

        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Level 1: 存在性判断 (2类)
        self.level1_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 2)
        )

        # Level 2: 状态分类 (3类)
        self.level2_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 3)
        )

        # Level 3: 细粒度分类 (7类)
        self.level3_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 7)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            Dict containing:
                - level1: 存在性预测 (B, 2)
                - level2: 状态预测 (B, 3)
                - level3: 细粒度预测 (B, 7)
        """
        feat = self.shared(x)  # (B, C, 1, 1)
        feat_flat = feat.flatten(1)  # (B, C)

        level1_pred = self.level1_head(feat_flat)
        level2_pred = self.level2_head(feat_flat)
        level3_pred = self.level3_head(feat_flat)

        return {
            "level1": level1_pred,
            "level2": level2_pred,
            "level3": level3_pred
        }


# ============================================================================
# 语义对齐模块
# ============================================================================

class SemanticAlignmentModule(nn.Module):
    """
    语义对齐模块

    确保视觉特征与语义标签的一致性
    使用KL散度损失 + 边界一致性损失
    """

    def __init__(self, in_channels: int, num_classes: int = 7):
        super().__init__()

        # 视觉特征投影
        self.visual_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, num_classes)
        )

        # 语义特征投影
        self.semantic_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, num_classes)
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_features: 视觉特征 (B, C, H, W)
            semantic_features: 语义特征 (B, C)

        Returns:
            visual_logits: 视觉logits (B, num_classes)
            semantic_logits: 语义logits (B, num_classes)
        """
        visual_logits = self.visual_proj(visual_features)
        semantic_logits = self.semantic_proj(semantic_features)

        return visual_logits, semantic_logits


# ============================================================================
# 完整的HD-DSAH检测头
# ============================================================================

class HDDSAH(nn.Module):
    """
    Hierarchical Decoupled Detection Head with Semantic Alignment

    层次化解耦检测头与语义对齐机制

    Args:
        in_channels: 输入特征通道数
        num_classes: 类别数 (默认7类)
        hidden_channels: 隐藏层通道数

    Features:
        1. 层次化三分类策略
        2. 解耦检测头 (分类/回归/置信度分离)
        3. 语义对齐模块
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 7,
        hidden_channels: int = 256
    ):
        super().__init__()

        self.num_classes = num_classes

        # 解耦检测头
        self.decoupled_head = DecoupledHead(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels
        )

        # 层次化分类头 (用于训练时辅助)
        self.hierarchical_head = HierarchicalClassificationHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels
        )

        # 语义对齐模块
        self.semantic_alignment = SemanticAlignmentModule(
            in_channels=in_channels,
            num_classes=num_classes
        )

    def forward(
        self,
        x: torch.Tensor,
        return_hierarchical: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of HD-DSAH

        Args:
            x: 输入特征 (B, C, H, W)
            return_hierarchical: 是否返回层次化预测

        Returns:
            Dict containing:
                - cls: 分类预测 (B, num_classes, H, W)
                - reg: 回归预测 (B, 4, H, W)
                - obj: 置信度预测 (B, 1, H, W)
                - visual_logits: 视觉语义logits (B, num_classes)
                - semantic_logits: 语义对齐logits (B, num_classes)
                - level1/2/3: 层次化预测 (可选)
        """
        # 解耦检测头
        cls_pred, reg_pred, obj_pred = self.decoupled_head(x)

        # 语义对齐
        semantic_feat = x.mean(dim=[2, 3])  # Global pooling
        visual_logits, semantic_logits = self.semantic_alignment(x, semantic_feat)

        outputs = {
            "cls": cls_pred,
            "reg": reg_pred,
            "obj": obj_pred,
            "visual_logits": visual_logits,
            "semantic_logits": semantic_logits
        }

        # 可选: 返回层次化预测
        if return_hierarchical:
            hierarchical_preds = self.hierarchical_head(x)
            outputs.update({
                "level1": hierarchical_preds["level1"],
                "level2": hierarchical_preds["level2"],
                "level3": hierarchical_preds["level3"]
            })

        return outputs


# ============================================================================
# 多尺度检测头 (支持P2/P3/P4/P5)
# ============================================================================

class HDDSAHMultiScale(nn.Module):
    """
    多尺度HD-DSAH检测头

    支持P2/P3/P4/P5多个尺度的检测
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 7,
        num_scales: int = 4  # P2, P3, P4, P5
    ):
        super().__init__()

        self.num_scales = num_scales
        self.num_classes = num_classes

        # 每个尺度的检测头
        self.heads = nn.ModuleList([
            HDDSAH(
                in_channels=in_channels,
                num_classes=num_classes
            ) for _ in range(num_scales)
        ])

    def forward(
        self,
        features: Tuple[torch.Tensor, ...],
        return_hierarchical: bool = False
    ) -> Dict[str, list]:
        """
        Args:
            features: 多尺度特征元组 (P2, P3, P4, P5)
            return_hierarchical: 是否返回层次化预测

        Returns:
            Dict containing:
                - cls: 各尺度分类预测列表
                - reg: 各尺度回归预测列表
                - obj: 各尺度置信度预测列表
                - visual_logits: 各尺度视觉logits列表
                - semantic_logits: 各尺度语义logits列表
        """
        outputs = {
            "cls": [],
            "reg": [],
            "obj": [],
            "visual_logits": [],
            "semantic_logits": []
        }

        if return_hierarchical:
            outputs["level1"] = []
            outputs["level2"] = []
            outputs["level3"] = []

        for i, feat in enumerate(features):
            head_out = self.heads[i](feat, return_hierarchical=return_hierarchical)

            outputs["cls"].append(head_out["cls"])
            outputs["reg"].append(head_out["reg"])
            outputs["obj"].append(head_out["obj"])
            outputs["visual_logits"].append(head_out["visual_logits"])
            outputs["semantic_logits"].append(head_out["semantic_logits"])

            if return_hierarchical:
                outputs["level1"].append(head_out["level1"])
                outputs["level2"].append(head_out["level2"])
                outputs["level3"].append(head_out["level3"])

        return outputs


# ============================================================================
# 损失函数
# ============================================================================

class HDDSAHLoss(nn.Module):
    """
    HD-DSAH损失函数

    包含:
    1. 分类损失 (Focal Loss)
    2. 回归损失 (CIoU Loss)
    3. 置信度损失 (BCE Loss)
    4. 语义对齐损失 (KL散度 + MSE)
    5. 层次化分类损失 (可选)
    """

    def __init__(
        self,
        num_classes: int = 7,
        use_hierarchical: bool = True,
        lambda_align: float = 0.1,
        lambda_hier: float = 0.05
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_hierarchical = use_hierarchical
        self.lambda_align = lambda_align
        self.lambda_hier = lambda_hier

        # Focal Loss参数
        self.alpha = 0.25
        self.gamma = 2.0

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Focal Loss for classification"""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

    def ciou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Complete IoU Loss"""
        # 计算IoU
        lt = torch.max(pred_boxes[..., :2], target_boxes[..., :2])
        rb = torch.min(pred_boxes[..., 2:], target_boxes[..., 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])

        union = pred_area + target_area - inter + 1e-7
        iou = inter / union

        # 中心点距离
        c_lt = torch.min(pred_boxes[..., :2], target_boxes[..., :2])
        c_rb = torch.max(pred_boxes[..., 2:], target_boxes[..., 2:])
        c_wh = (c_rb - c_lt).clamp(min=0)
        c_diag = c_wh[..., 0] ** 2 + c_wh[..., 1] ** 2 + 1e-7

        pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
        target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
        center_dist = (pred_center - target_center).pow(2).sum(dim=-1) + 1e-7

        # 长宽比一致性
        pred_wh = (pred_boxes[..., 2:] - pred_boxes[..., :2]).clamp(min=1e-6)
        target_wh = (target_boxes[..., 2:] - target_boxes[..., :2]).clamp(min=1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(pred_wh[..., 0] / pred_wh[..., 1]) -
            torch.atan(target_wh[..., 0] / target_wh[..., 1]), 2
        )

        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - (center_dist / c_diag) - alpha * v
        return (1 - ciou).mean()

    def semantic_alignment_loss(
        self,
        visual_logits: torch.Tensor,
        semantic_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        语义对齐损失

        包含KL散度和边界一致性
        """
        # KL散度损失
        visual_prob = F.softmax(visual_logits / 1.0, dim=-1)
        semantic_prob = F.softmax(semantic_logits / 1.0, dim=-1)

        kl_loss = F.kl_div(
            F.log_softmax(visual_logits, dim=-1),
            semantic_prob,
            reduction='batchmean'
        )

        # 分类损失 (与目标标签对齐)
        target_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        cls_loss_visual = F.cross_entropy(visual_logits, targets)
        cls_loss_semantic = F.cross_entropy(semantic_logits, targets)

        return kl_loss + cls_loss_visual + cls_loss_semantic

    def hierarchical_loss(
        self,
        level1_pred: torch.Tensor,
        level2_pred: torch.Tensor,
        level3_pred: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """层次化分类损失"""
        batch_size = targets.shape[0]

        # 根据target构建层次化标签
        level1_targets = torch.zeros(batch_size, 2, device=targets.device)
        level2_targets = torch.zeros(batch_size, 3, device=targets.device)
        level3_targets = torch.zeros(batch_size, 7, device=targets.device)

        for i, t in enumerate(targets):
            level_info = ManholeStatusHierarchy.CLASS_TO_LEVEL[t.item()]
            level1_targets[i, level_info[0]] = 1
            level2_targets[i, level_info[1]] = 1
            level3_targets[i, t] = 1

        level1_loss = F.cross_entropy(level1_pred, level1_targets.argmax(dim=-1))
        level2_loss = F.cross_entropy(level2_pred, level2_targets.argmax(dim=-1))
        level3_loss = F.cross_entropy(level3_pred, level3_targets.argmax(dim=-1))

        return level1_loss + level2_loss + level3_loss

    def forward(
        self,
        predictions: Dict,
        targets: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            predictions: 模型预测
            targets: 目标标签

        Returns:
            Dict containing:
                - total_loss: 总损失
                - cls_loss: 分类损失
                - reg_loss: 回归损失
                - obj_loss: 置信度损失
                - align_loss: 语义对齐损失
                - hier_loss: 层次化损失 (可选)
        """
        # 分类损失
        cls_loss = self.focal_loss(
            predictions["cls"],
            targets["cls_labels"]
        )

        # 回归损失
        reg_loss = self.ciou_loss(
            predictions["reg"],
            targets["boxes"]
        )

        # 置信度损失
        obj_loss = F.binary_cross_entropy_with_logits(
            predictions["obj"],
            targets["obj_labels"]
        )

        # 语义对齐损失
        align_loss = self.semantic_alignment_loss(
            predictions["visual_logits"],
            predictions["semantic_logits"],
            targets["cls_labels"].argmax(dim=-1) if targets["cls_labels"].dim() > 1 else targets["cls_labels"]
        )

        # 层次化损失 (可选)
        hier_loss = torch.tensor(0.0, device=cls_loss.device)
        if self.use_hierarchical and "level1" in predictions:
            # 取第一个尺度的层次化预测
            hier_loss = self.hierarchical_loss(
                predictions["level1"][0] if isinstance(predictions["level1"], list) else predictions["level1"],
                predictions["level2"][0] if isinstance(predictions["level2"], list) else predictions["level2"],
                predictions["level3"][0] if isinstance(predictions["level3"], list) else predictions["level3"],
                targets["cls_labels"].argmax(dim=-1) if targets["cls_labels"].dim() > 1 else targets["cls_labels"]
            )

        # 总损失
        total_loss = (
            cls_loss +
            reg_loss +
            obj_loss +
            self.lambda_align * align_loss +
            self.lambda_hier * hier_loss
        )

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "obj_loss": obj_loss,
            "align_loss": align_loss,
            "hier_loss": hier_loss
        }


# ============================================================================
# 工具函数
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HD-DSAH Module Test")
    print("=" * 60)

    batch_size = 2
    in_channels = 256
    height, width = 80, 80  # P3层分辨率

    # 测试HDDSAH检测头
    print("\n1. Testing HDDSAH Detection Head")
    hdsah = HDDSAH(in_channels=in_channels, num_classes=7)
    print(f"   Parameters: {count_parameters(hdsah):,}")

    x = torch.randn(batch_size, in_channels, height, width)
    outputs = hdsah(x, return_hierarchical=True)

    print(f"   Input shape: {x.shape}")
    print(f"   Classification output: {outputs['cls'].shape}")
    print(f"   Regression output: {outputs['reg'].shape}")
    print(f"   Objectness output: {outputs['obj'].shape}")
    print(f"   Visual logits: {outputs['visual_logits'].shape}")
    print(f"   Semantic logits: {outputs['semantic_logits'].shape}")
    print(f"   Level1 (existence): {outputs['level1'].shape}")
    print(f"   Level2 (status): {outputs['level2'].shape}")
    print(f"   Level3 (fine-grained): {outputs['level3'].shape}")

    # 测试多尺度检测头
    print("\n2. Testing Multi-Scale HDDSAH")
    multi_scale_head = HDDSAHMultiScale(in_channels=256, num_classes=7, num_scales=4)
    print(f"   Parameters: {count_parameters(multi_scale_head):,}")

    # 模拟P2, P3, P4, P5特征
    features = (
        torch.randn(batch_size, 256, 160, 160),  # P2
        torch.randn(batch_size, 256, 80, 80),    # P3
        torch.randn(batch_size, 256, 40, 40),    # P4
        torch.randn(batch_size, 256, 20, 20),    # P5
    )

    outputs = multi_scale_head(features, return_hierarchical=True)

    print(f"   P2 classification output: {outputs['cls'][0].shape}")
    print(f"   P3 classification output: {outputs['cls'][1].shape}")
    print(f"   P4 classification output: {outputs['cls'][2].shape}")
    print(f"   P5 classification output: {outputs['cls'][3].shape}")

    # 测试损失函数
    print("\n3. Testing HDDSAH Loss")
    loss_fn = HDDSAHLoss(num_classes=7, use_hierarchical=True)

    # 模拟预测
    predictions = {
        "cls": torch.randn(batch_size, 7, height, width),
        "reg": torch.rand(batch_size, 4, height, width) * 100,
        "obj": torch.randn(batch_size, 1, height, width),
        "visual_logits": torch.randn(batch_size, 7),
        "semantic_logits": torch.randn(batch_size, 7),
        "level1": torch.randn(batch_size, 2),
        "level2": torch.randn(batch_size, 3),
        "level3": torch.randn(batch_size, 7),
    }

    # 模拟目标
    targets = {
        "cls_labels": torch.randint(0, 7, (batch_size,)),
        "boxes": torch.rand(batch_size, 4) * 100,
        "obj_labels": torch.randint(0, 2, (batch_size, 1, height, width)).float(),
    }

    losses = loss_fn(predictions, targets)
    print(f"   Total loss: {losses['total_loss'].item():.4f}")
    print(f"   Classification loss: {losses['cls_loss'].item():.4f}")
    print(f"   Regression loss: {losses['reg_loss'].item():.4f}")
    print(f"   Objectness loss: {losses['obj_loss'].item():.4f}")
    print(f"   Alignment loss: {losses['align_loss'].item():.4f}")
    print(f"   Hierarchical loss: {losses['hier_loss'].item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
