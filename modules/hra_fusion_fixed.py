"""
HRA-Fusion Module: High-Resolution Adaptive Fusion Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ChannelAttention(nn.Module):
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
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class LightweightTransformer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x + identity


class HRAFusion(nn.Module):
    def __init__(self, in_channels: int = 256, out_channels: int = 256, reduction: int = 16):
        super().__init__()
        self.branch_a_conv3x3 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3)
        self.branch_a_conv5x5 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=5)
        self.branch_a_cbam = CBAM(in_channels, reduction)
        self.branch_b_global = LightweightTransformer(in_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x_p2: torch.Tensor, x_p3: Optional[torch.Tensor] = None,
                x_p4: Optional[torch.Tensor] = None, x_p5: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat_local_3x3 = self.branch_a_conv3x3(x_p2)
        feat_local_5x5 = self.branch_a_conv5x5(x_p2)
        feat_local = (feat_local_3x3 + feat_local_5x5) / 2
        feat_local = self.branch_a_cbam(feat_local)
        feat_global = self.branch_b_global(x_p2)
        alpha = torch.sigmoid(feat_local.mean(dim=[2, 3], keepdim=True))
        beta = torch.sigmoid(feat_global.mean(dim=[2, 3], keepdim=True))
        sum_weights = alpha + beta + 1e-8
        alpha = alpha / sum_weights
        beta = beta / sum_weights
        feat_fused = alpha * feat_local + beta * feat_global
        if x_p3 is not None:
            x_p3_up = F.interpolate(x_p3, size=x_p2.shape[2:], mode='nearest')
            feat_fused = feat_fused + x_p3_up * 0.3
        if x_p4 is not None:
            x_p4_up = F.interpolate(x_p4, size=x_p2.shape[2:], mode='nearest')
            feat_fused = feat_fused + x_p4_up * 0.15
        if x_p5 is not None:
            x_p5_up = F.interpolate(x_p5, size=x_p2.shape[2:], mode='nearest')
            feat_fused = feat_fused + x_p5_up * 0.05
        feat_cat = torch.cat([feat_local, feat_global], dim=1)
        feat_fused = self.fusion_conv(feat_cat) + feat_fused
        out = self.out_conv(feat_fused)
        out = self.out_bn(out)
        return out


if __name__ == "__main__":
    print("Testing HRA-Fusion Module...")
    hra = HRAFusion(in_channels=256, out_channels=256)
    x = torch.randn(2, 256, 80, 80)
    out = hra(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in hra.parameters()):,}")
    print("Test passed!")
