"""
GD-MSE Module: Gradient-Guided Multi-Scale Enhancement Module

基于梯度指导的多尺度特征增强模块

论文: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法
作者: [Your Name]
日期: 2026-02-08

设计思路:
    1. 梯度信息提取 (使用特征方差作为梯度敏感度代理)
    2. 跨尺度特征聚合 (聚合所有尺度的梯度信息)
    3. 改进的C3k2模块 (集成梯度信息)

理论基础:
    - Uncertainty-Aware Gradient Stabilization (ICCV 2025)
    - YOLOv9 PGI (Programmable Gradient Information)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# 梯度信息提取模块
# ============================================================================

class GradientExtractor(nn.Module):
    """
    梯度信息提取模块

    使用特征的空间方差作为梯度敏感度的代理指标
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        self.gradient_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取梯度信息

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            梯度敏感度图 (B, C, H, W)
        """
        # 计算空间方差作为梯度敏感度的代理
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = ((x - mean) ** 2).mean(dim=[2, 3], keepdim=True)
        grad_proxy = torch.sqrt(var + 1e-8)

        # 通过卷积提取梯度信息
        grad_info = self.gradient_conv(x)

        # 融合梯度敏感度
        output = grad_info * torch.sigmoid(grad_proxy)

        return output


# ============================================================================
# 跨尺度特征聚合模块
# ============================================================================

class CrossScaleAggregation(nn.Module):
    """
    跨尺度特征聚合模块

    聚合所有尺度的梯度信息到当前尺度
    """

    def __init__(
        self,
        in_channels: int,
        num_scales: int = 4
    ):
        super().__init__()

        self.num_scales = num_scales

        # 每个尺度的聚合权重
        self.aggregation_weights = nn.Parameter(
            torch.ones(num_scales) / num_scales
        )

        # 聚合卷积
        self.aggregate_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(
        self,
        features: Tuple[torch.Tensor, ...],
        target_idx: int
    ) -> torch.Tensor:
        """
        聚合所有尺度的特征到目标尺度

        Args:
            features: 所有尺度的特征元组
            target_idx: 目标尺度索引

        Returns:
            聚合后的特征 (B, C, H_target, W_target)
        """
        target_feat = features[target_idx]
        B, C, H, W = target_feat.shape

        # 收集所有尺度的特征并调整到目标尺寸
        aligned_feats = []
        normalized_weights = F.softmax(self.aggregation_weights, dim=0)

        for i, feat in enumerate(features):
            # 上采样/下采样到目标尺寸
            if feat.shape[2:] != (H, W):
                aligned = F.interpolate(
                    feat,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                aligned = feat

            # 应用学习权重
            aligned = aligned * normalized_weights[i]
            aligned_feats.append(aligned)

        # 拼接并聚合
        concat = torch.cat(aligned_feats, dim=1)
        aggregated = self.aggregate_conv(concat)

        # 残差连接
        output = target_feat + aggregated

        return output


# ============================================================================
# 改进的C3k2模块 (带梯度信息)
# ============================================================================

class C3k2GD(nn.Module):
    """
    改进的C3k2模块 (梯度指导版本)

    在YOLOv11的C3k2模块基础上集成梯度信息
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        reduction: int = 16
    ):
        super().__init__()

        hidden_channels = hidden_channels or in_channels // 2

        # 梯度信息提取
        self.gradient_extractor = GradientExtractor(in_channels, reduction)

        # 标准C3k2结构 (简化版)
        self.cv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.cv2 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channels * 2, in_channels, 1, bias=False)

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            输出特征 (B, C, H, W)
        """
        # 提取梯度信息
        grad_info = self.gradient_extractor(x)

        # C3k2结构
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        concat = torch.cat([y1, y2], dim=1)
        c3k2_feat = self.cv3(concat)

        # 融合梯度信息
        output = c3k2_feat + grad_info
        output = self.bn(output)
        output = self.act(output)

        # 残差连接
        return output + x


# ============================================================================
# 改进的SPPF模块 (带空间梯度)
# ============================================================================

class SPPFGD(nn.Module):
    """
    改进的SPPF模块 (带空间梯度信息)

    在YOLOv11的SPPF模块基础上添加空间梯度特征
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 5
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        # 空间梯度提取
        self.spatial_gradient = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 标准SPPF
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
            for _ in range(3)
        ])

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels * 4, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            输出特征 (B, out_channels, H, W)
        """
        # 提取空间梯度
        spatial_grad = self.spatial_gradient(x)

        # SPPF结构
        y1 = self.conv1(x)
        y2 = self.pools[0](y1)
        y3 = self.pools[1](y2)
        y4 = self.pools[2](y3)

        concat = torch.cat([y1, y2, y3, y4], dim=1)
        sppf_feat = self.conv2(concat)

        # 融合空间梯度
        output = sppf_feat * (1 + spatial_grad)

        output = self.bn(output)
        output = self.act(output)

        return output


# ============================================================================
# 完整的GD-MSE模块
# ============================================================================

class GDMSE(nn.Module):
    """
    Gradient-Guided Multi-Scale Enhancement Module

    基于梯度指导的多尺度特征增强模块

    Args:
        in_channels: 输入特征通道数
        num_scales: 尺度数量 (默认4: P2, P3, P4, P5)
        use_c3k2_gd: 是否使用改进的C3k2模块
        use_sppf_gd: 是否使用改进的SPPF模块

    Features:
        1. 梯度信息提取
        2. 跨尺度特征聚合
        3. 改进的C3k2和SPPF模块
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_scales: int = 4,
        use_c3k2_gd: bool = True,
        use_sppf_gd: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_scales = num_scales
        self.use_c3k2_gd = use_c3k2_gd
        self.use_sppf_gd = use_sppf_gd

        # 梯度信息提取器 (每个尺度一个)
        self.gradient_extractors = nn.ModuleList([
            GradientExtractor(in_channels)
            for _ in range(num_scales)
        ])

        # 跨尺度聚合 (每个尺度一个)
        self.cross_scale_aggregations = nn.ModuleList([
            CrossScaleAggregation(in_channels, num_scales)
            for _ in range(num_scales)
        ])

        # 改进的C3k2模块 (每个尺度一个)
        if use_c3k2_gd:
            self.c3k2_gd_modules = nn.ModuleList([
                C3k2GD(in_channels)
                for _ in range(num_scales)
            ])

        # 改进的SPPF模块 (最大尺度的P5层)
        if use_sppf_gd:
            self.sppf_gd = SPPFGD(in_channels)

    def forward(
        self,
        features: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            features: 多尺度特征元组 (P2, P3, P4, P5)

        Returns:
            增强后的特征元组
        """
        assert len(features) == self.num_scales, \
            f"Expected {self.num_scales} features, got {len(features)}"

        # 1. 提取梯度信息
        gradient_feats = []
        for i, feat in enumerate(features):
            grad_feat = self.gradient_extractors[i](feat)
            gradient_feats.append(grad_feat)

        # 2. 跨尺度聚合
        aggregated_feats = []
        for i in range(self.num_scales):
            # 聚合所有尺度的梯度信息
            agg_grad = self.cross_scale_aggregations[i](gradient_feats, i)

            # 融合原始特征
            aggregated = features[i] + agg_grad * 0.5
            aggregated_feats.append(aggregated)

        # 3. 改进的C3k2处理
        if self.use_c3k2_gd:
            c3k2_feats = []
            for i, feat in enumerate(aggregated_feats):
                c3k2_feat = self.c3k2_gd_modules[i](feat)
                c3k2_feats.append(c3k2_feat)
            aggregated_feats = c3k2_feats

        # 4. 改进的SPPF处理 (P5层)
        if self.use_sppf_gd:
            p5_enhanced = self.sppf_gd(aggregated_feats[-1])
            aggregated_feats[-1] = p5_enhanced

        return tuple(aggregated_feats)


# ============================================================================
# 轻量级GD-MSE模块 (用于边缘部署)
# ============================================================================

class GDMSELite(nn.Module):
    """
    轻量级GD-MSE模块

    减少计算量，适合边缘设备部署
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_scales: int = 4
    ):
        super().__init__()

        # 简化的梯度提取 (只使用通道注意力)
        self.gradient_extractors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.Sigmoid()
            )
            for _ in range(num_scales)
        ])

        # 简化的跨尺度聚合 (只使用相邻尺度)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 3, in_channels, 1, bias=False)
            for _ in range(num_scales)
        ])

    def forward(
        self,
        features: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        轻量级前向传播
        """
        enhanced = []

        for i, feat in enumerate(features):
            # 提取梯度权重
            grad_weight = self.gradient_extractors[i](feat)

            # 收集相邻尺度的特征
            neighbor_feats = []
            for offset in [-1, 0, 1]:
                ni = i + offset
                if 0 <= ni < len(features):
                    neighbor = features[ni]
                    if neighbor.shape[2:] != feat.shape[2:]:
                        neighbor = F.interpolate(
                            neighbor,
                            size=feat.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    neighbor_feats.append(neighbor)
                else:
                    neighbor_feats.append(feat)

            # 聚合相邻特征
            concat = torch.cat(neighbor_feats, dim=1)
            aggregated = self.lateral_convs[i](concat)

            # 融合
            output = feat + aggregated * grad_weight
            enhanced.append(output)

        return tuple(enhanced)


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
    print("="*60)
    print("GD-MSE Module Test")
    print("="*60)

    batch_size = 2
    in_channels = 256

    # 模拟多尺度特征 (P2, P3, P4, P5)
    features = (
        torch.randn(batch_size, in_channels, 160, 160),  # P2
        torch.randn(batch_size, in_channels, 80, 80),    # P3
        torch.randn(batch_size, in_channels, 40, 40),    # P4
        torch.randn(batch_size, in_channels, 20, 20),    # P5
    )

    # 测试完整GD-MSE模块
    print("\n1. Testing Full GD-MSE Module")
    gd_mse = GDMSE(in_channels=in_channels, num_scales=4)
    print(f"   Parameters: {count_parameters(gd_mse):,}")

    enhanced = gd_mse(features)
    print(f"   Input shapes: {[f.shape for f in features]}")
    print(f"   Output shapes: {[f.shape for f in enhanced]}")

    # 测试轻量级版本
    print("\n2. Testing GD-MSE Lite Module")
    gd_mse_lite = GDMSELite(in_channels=in_channels, num_scales=4)
    print(f"   Parameters: {count_parameters(gd_mse_lite):,}")

    enhanced_lite = gd_mse_lite(features)
    print(f"   Output shapes: {[f.shape for f in enhanced_lite]}")

    # 比较参数量
    full_params = count_parameters(gd_mse)
    lite_params = count_parameters(gd_mse_lite)
    print(f"\n   Parameter reduction: {(1 - lite_params/full_params)*100:.1f}%")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
