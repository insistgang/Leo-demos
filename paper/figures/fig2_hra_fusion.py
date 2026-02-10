import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# 颜色定义
color_input = '#E8E8E8'
color_cnn = '#4A90D9'  # 蓝色 - CNN分支
color_trans = '#9B59B6'  # 紫色 - Transformer分支
color_cbam = '#E67E22'  # 橙色 - CBAM
color_fusion = '#27AE60'  # 绿色 - 融合
color_arrow = '#2C3E50'

def draw_box(ax, x, y, width, height, color, text, fontsize=9, text_color='white', radius=0.05):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle=f"round,pad=0.02,rounding_size={radius}",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=text_color, fontweight='bold', wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=color_arrow, lw=1.5):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def draw_split_arrow(ax, x, y, y1, y2, color=color_arrow):
    """绘制分叉箭头"""
    # 主干
    ax.plot([x, x+0.3], [y, y], color=color, lw=1.5)
    # 分叉
    ax.plot([x+0.3, x+0.6], [y, y1], color=color, lw=1.5)
    ax.plot([x+0.3, x+0.6], [y, y2], color=color, lw=1.5)
    # 箭头
    ax.annotate('', xy=(x+0.6, y1), xytext=(x+0.3, (y+y1)/2),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax.annotate('', xy=(x+0.6, y2), xytext=(x+0.3, (y+y2)/2),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# ============ 绘制输入 ============
draw_box(ax, 1.2, 4, 1.2, 0.7, color_input, 'Input\n$F_2$', fontsize=10, text_color='black')

# ============ 分叉箭头 ============
draw_split_arrow(ax, 1.8, 4, 5.5, 2.5)

# ============ 分支1: CNN局部特征 (上方) ============
# 分支标题
ax.text(3.5, 6.2, 'Branch 1: CNN Local Features', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=color_cnn)

# DWConv 3x3
draw_box(ax, 3.2, 5.5, 1.4, 0.6, color_cnn, 'DWConv\n3×3, s=1', fontsize=8)
draw_arrow(ax, 2.4, 5.5, 2.5, 5.5)

# DWConv 5x5
draw_box(ax, 5.0, 5.5, 1.4, 0.6, color_cnn, 'DWConv\n5×5, s=1', fontsize=8)
draw_arrow(ax, 3.9, 5.5, 4.3, 5.5)

# F_local
draw_box(ax, 6.8, 5.5, 1.0, 0.6, color_cnn, '$F_{local}$', fontsize=10)
draw_arrow(ax, 5.7, 5.5, 6.3, 5.5)

# ============ 分支2: Transformer全局特征 (下方) ============
# 分支标题
ax.text(3.5, 1.8, 'Branch 2: Transformer Global Features', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=color_trans)

# LightTransformer
draw_box(ax, 3.5, 2.5, 2.0, 0.7, color_trans, 'LightTransformer\n(MHSA + FFN)', fontsize=8)
draw_arrow(ax, 2.4, 2.5, 2.5, 2.5)

# F_global
draw_box(ax, 5.8, 2.5, 1.0, 0.6, color_trans, '$F_{global}$', fontsize=10)
draw_arrow(ax, 4.5, 2.5, 5.3, 2.5)

# ============ CBAM注意力模块 ============
# CBAM主框
cbam_box = FancyBboxPatch((6.8, 3.2), 2.0, 1.6,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#FEF5E7', edgecolor=color_cbam, linewidth=2)
ax.add_patch(cbam_box)
ax.text(7.8, 4.6, 'CBAM Attention', ha='center', va='center', 
        fontsize=10, fontweight='bold', color=color_cbam)

# 通道注意力
draw_box(ax, 7.3, 4.0, 1.4, 0.45, color_cbam, 'Channel Attention', fontsize=8)
# 空间注意力
draw_box(ax, 7.3, 3.4, 1.4, 0.45, color_cbam, 'Spatial Attention', fontsize=8)

# CBAM输入箭头
draw_arrow(ax, 7.3, 5.5, 7.3, 4.5, color=color_arrow)
ax.plot([6.8, 7.3], [5.5, 5.5], color=color_arrow, lw=1.5)
ax.plot([7.3, 7.3], [4.5, 4.5], color=color_arrow, lw=1.5)

# 从F_global到CBAM
draw_arrow(ax, 6.3, 2.5, 7.0, 2.5, color=color_arrow)
ax.plot([7.0, 7.0], [2.5, 3.2], color=color_arrow, lw=1.5)
ax.plot([7.0, 7.3], [3.2, 3.4], color=color_arrow, lw=1.5)

# 输出权重α, β
ax.text(8.8, 4.0, r'$\alpha$', ha='center', va='center', fontsize=12, fontweight='bold', color=color_cbam)
ax.text(8.8, 3.4, r'$\beta$', ha='center', va='center', fontsize=12, fontweight='bold', color=color_cbam)
ax.annotate('', xy=(9.0, 4.0), xytext=(8.5, 4.0),
            arrowprops=dict(arrowstyle='->', color=color_cbam, lw=1.5))
ax.annotate('', xy=(9.0, 3.4), xytext=(8.5, 3.4),
            arrowprops=dict(arrowstyle='->', color=color_cbam, lw=1.5))

# ============ 融合模块 ============
# 融合框
fusion_box = FancyBboxPatch((8.8, 2.8), 1.0, 1.8,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='#E8F8F5', edgecolor=color_fusion, linewidth=2)
ax.add_patch(fusion_box)
ax.text(9.3, 4.3, 'Fusion', ha='center', va='center', 
        fontsize=10, fontweight='bold', color=color_fusion)

# 融合公式
ax.text(9.3, 3.5, r'$F_{fused}$', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(9.3, 3.0, r'$=\alpha F_{local}$', ha='center', va='center', fontsize=9)
ax.text(9.3, 2.7, r'$+\beta F_{global}$', ha='center', va='center', fontsize=9)

# F_local到融合
draw_arrow(ax, 7.3, 5.5, 8.8, 4.2, color=color_arrow)

# F_global到融合
draw_arrow(ax, 6.3, 2.5, 8.8, 3.2, color=color_arrow)

# ============ 输出 ============
draw_box(ax, 9.3, 1.2, 1.2, 0.6, '#FADBD8', '$F_{fused}$', fontsize=11, text_color='#C0392B')
draw_arrow(ax, 9.3, 2.0, 9.3, 1.5, color=color_arrow)

# ============ 标题 ============
ax.text(5, 7.4, '图2  HRA-Fusion模块结构', ha='center', va='center', 
        fontsize=14, fontweight='bold')

# ============ 图例 ============
legend_elements = [
    mpatches.Patch(facecolor=color_cnn, edgecolor='black', label='CNN Branch (Local)'),
    mpatches.Patch(facecolor=color_trans, edgecolor='black', label='Transformer Branch (Global)'),
    mpatches.Patch(facecolor=color_cbam, edgecolor='black', label='CBAM Attention'),
    mpatches.Patch(facecolor=color_fusion, edgecolor='black', label='Fusion')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9, 
          framealpha=0.9, edgecolor='gray')

# ============ 添加说明文字 ============
ax.text(0.3, 0.5, 'DWConv: Depthwise Separable Convolution  |  MHSA: Multi-Head Self-Attention  |  FFN: Feed-Forward Network', 
        ha='left', va='center', fontsize=8, style='italic', color='gray')

plt.tight_layout()

# 使用原始字符串或正斜杠处理Windows路径
output_dir = r'D:\jglw\yolov11-manhole-detection\paper\figures'
png_path = os.path.join(output_dir, 'fig2_hra_fusion.png')
pdf_path = os.path.join(output_dir, 'fig2_hra_fusion.pdf')

plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
print("图2 HRA-Fusion模块结构图已生成!")
print("输出文件:")
print(f"  - PNG: {png_path}")
print(f"  - PDF: {pdf_path}")
