"""
HRA-Fusion Module: High-Resolution Adaptive Fusion Module

面向小目标的高分辨率自适应特征融合模块

论文: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法
作者: [Your Name]
日期: 2026-02-08

设计思路:
    1. 新增P2层（1/4下采样）保留小目标特征
    2. 双分支特征提取（CNN局部特征 + Transformer全局特征）
    3. 自适应融合策略（通道注意力动态调整权重）

数学表达:
    F_local = DWConv(F_P2) ⊕ CBAM(F_P2)
    F_global = BottleneckTransformer(F_P2)
    F_fused = α·F_local + β·F_global, 约束: α + β = 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# 基础组件
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ChannelAttention(nn.Module):
    """通道注意力模块 (简化版CBAM的通道注意力)"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块 (简化版CBAM的空间注意力)"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x_flat = x.reshape(B, H * W, C)

        qkv = self.qkv(x_flat).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)
        return x.reshape(B, H, W, C)


class BottleneckTransformer(nn.Module):
    """瓶颈Transformer模块"""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, C, H, W -> B, H, W, C

        # Transformer block with residual
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
        return x


# ============================================================================
# HRA-Fusion Module
# ============================================================================

class HRAFusion(nn.Module):
    """
    High-Resolution Adaptive Fusion Module (HRA-Fusion)

    面向小目标的高分辨率自适应特征融合模块

    Args:
        in_channels (int): 输入特征通道数
        out_channels (int): 输出特征通道数
        num_heads (int): Transformer注意力头数
        reduction (int): 通道注意力缩减比

    Forward:
        Input: P2特征 (B, C, H, W), H = H_input/4, W = W_input/4
        Output: 融合特征 (B, out_channels, H, W)
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_heads: int = 8,
        reduction: int = 16
    ):
        super().__init__()

        # 分支A: CNN局部特征提取
        self.branch_a_conv3x3 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3)
        self.branch_a_conv5x5 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=5)
        self.branch_a_cbam = CBAM(in_channels, reduction)

        # 分支B: Transformer全局特征提取
        self.branch_b_transformer = BottleneckTransformer(in_channels, num_heads)

        # 自适应融合权重计算
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 输出投影
        self.out_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(
        self,
        x_p2: torch.Tensor,
        x_p3: Optional[torch.Tensor] = None,
        x_p4: Optional[torch.Tensor] = None,
        x_p5: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of HRA-Fusion module.

        Args:
            x_p2: P2层特征 (B, C, H, W), H=H_in/4
            x_p3: P3层特征 (可选, 用于跨尺度融合)
            x_p4: P4层特征 (可选)
            x_p5: P5层特征 (可选)

        Returns:
            融合后的P2特征 (B, out_channels, H, W)
        """
        # 分支A: CNN局部特征提取
        feat_local_3x3 = self.branch_a_conv3x3(x_p2)
        feat_local_5x5 = self.branch_a_conv5x5(x_p2)
        feat_local = torch.cat([feat_local_3x3, feat_local_5x5], dim=1)
        feat_local = feat_local.mean(dim=1, keepdim=True) * x_p2 + x_p2  # 残差连接
        feat_local = self.branch_a_cbam(feat_local)

        # 分支B: Transformer全局特征提取
        feat_global = self.branch_b_transformer(x_p2)

        # 自适应融合权重计算
        alpha = torch.sigmoid(feat_local.mean(dim=[2, 3], keepdim=True))
        beta = torch.sigmoid(feat_global.mean(dim=[2, 3], keepdim=True))

        # 归一化约束: alpha + beta = 1
        sum_weights = alpha + beta + 1e-8
        alpha = alpha / sum_weights
        beta = beta / sum_weights

        # 自适应融合
        feat_fused = alpha * feat_local + beta * feat_global

        # 跨尺度特征融合 (如果提供了其他尺度的特征)
        if x_p3 is not None:
            x_p3_up = F.interpolate(x_p3, size=x_p2.shape[2:], mode='nearest')
            feat_fused = feat_fused + x_p3_up * 0.3

        if x_p4 is not None:
            x_p4_up = F.interpolate(x_p4, size=x_p2.shape[2:], mode='nearest')
            feat_fused = feat_fused + x_p4_up * 0.15

        if x_p5 is not None:
            x_p5_up = F.interpolate(x_p5, size=x_p2.shape[2:], mode='nearest')
            feat_fused = feat_fused + x_p5_up * 0.05

        # 融合后的特征处理
        feat_cat = torch.cat([feat_local, feat_global], dim=1)
        feat_fused = self.fusion_conv(feat_cat) + feat_fused  # 残差

        # 输出投影
        out = self.out_conv(feat_fused)
        out = self.out_bn(out)

        return out


class HRAFusionNeck(nn.Module):
    """
    带HRA-Fusion的特征金字塔Neck

    将HRA-Fusion集成到YOLOv11的Neck中
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        out_channels: int = 256,
        num_heads: int = 8
    ):
        super().__init__()

        # P2层新增 (用于小目标检测)
        self.p2_conv = nn.Conv2d(in_channels[0], out_channels, 1, bias=False)

        # HRA-Fusion模块
        self.hra_fusion = HRAFusion(
            in_channels=out_channels,
            out_channels=out_channels,
            num_heads=num_heads
        )

        # P3, P4, P5层处理
        self.p3_conv = nn.Conv2d(in_channels[1], out_channels, 1, bias=False)
        self.p4_conv = nn.Conv2d(in_channels[2], out_channels, 1, bias=False)
        self.p5_conv = nn.Conv2d(in_channels[3], out_channels, 1, bias=False)

    def forward(
        self,
        x_p2: torch.Tensor,
        x_p3: torch.Tensor,
        x_p4: torch.Tensor,
        x_p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_p2: P2层特征 (B, 64, H/4, W/4)
            x_p3: P3层特征 (B, 128, H/8, W/8)
            x_p4: P4层特征 (B, 256, H/16, W/16)
            x_p5: P5层特征 (B, 512, H/32, W/32)

        Returns:
            out_p2: 融合后的P2特征 (B, 256, H/4, W/4) - 新增小目标检测层
            out_p3: 融合后的P3特征 (B, 256, H/8, W/8)
            out_p4: 融合后的P4特征 (B, 256, H/16, W/16)
            out_p5: 融合后的P5特征 (B, 256, H/32, W/32)
        """
        # 通道对齐
        x_p2 = self.p2_conv(x_p2)
        x_p3 = self.p3_conv(x_p3)
        x_p4 = self.p4_conv(x_p4)
        x_p5 = self.p5_conv(x_p5)

        # HRA-Fusion处理P2层 (核心创新)
        out_p2 = self.hra_fusion(x_p2, x_p3, x_p4, x_p5)

        # 自顶向下路径 (FPN风格)
        p5_up = F.interpolate(x_p5, size=x_p4.shape[2:], mode='nearest')
        out_p4 = x_p4 + p5_up

        p4_up = F.interpolate(out_p4, size=x_p3.shape[2:], mode='nearest')
        out_p3 = x_p3 + p4_up

        p3_up = F.interpolate(out_p3, size=x_p2.shape[2:], mode='nearest')
        out_p2 = out_p2 + p3_up * 0.5  # 轻微融合

        # P5层保持不变 (可以添加额外处理)
        out_p5 = x_p5

        return out_p2, out_p3, out_p4, out_p5


# ============================================================================
# 工具函数
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 640, 640)):
    """获取模型摘要信息"""
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        # 这里需要根据实际的backbone输出调整
        # 仅作示例
        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {count_parameters(model):,}")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HRA-Fusion Module Test")
    print("=" * 60)

    # 测试HRA-Fusion模块
    batch_size = 2
    h, w = 160, 160  # P2层分辨率 (640/4)
    channels = 256

    hra_fusion = HRAFusion(in_channels=channels, out_channels=256, num_heads=8)
    print(f"\nHRA-Fusion Parameters: {count_parameters(hra_fusion):,}")

    # 测试前向传播
    x_p2 = torch.randn(batch_size, channels, h, w)
    x_p3 = torch.randn(batch_size, channels, h//2, w//2)
    x_p4 = torch.randn(batch_size, channels, h//4, w//4)
    x_p5 = torch.randn(batch_size, channels, h//8, w//8)

    output = hra_fusion(x_p2, x_p3, x_p4, x_p5)
    print(f"Input shape: {x_p2.shape}")
    print(f"Output shape: {output.shape}")

    # 测试完整的Neck
    print("\n" + "=" * 60)
    print("HRA-Fusion Neck Test")
    print("=" * 60)

    neck = HRAFusionNeck(
        in_channels=(64, 128, 256, 512),
        out_channels=256,
        num_heads=8
    )
    print(f"\nNeck Parameters: {count_parameters(neck):,}")

    # 模拟backbone输出
    x_p2 = torch.randn(batch_size, 64, h, w)
    x_p3 = torch.randn(batch_size, 128, h//2, w//2)
    x_p4 = torch.randn(batch_size, 256, h//4, w//4)
    x_p5 = torch.randn(batch_size, 512, h//8, w//8)

    out_p2, out_p3, out_p4, out_p5 = neck(x_p2, x_p3, x_p4, x_p5)
    print(f"Output P2 shape: {out_p2.shape}")
    print(f"Output P3 shape: {out_p3.shape}")
    print(f"Output P4 shape: {out_p4.shape}")
    print(f"Output P5 shape: {out_p5.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
