import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Set up Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Color definitions
colors = {
    'input_output': '#808080',  # Gray
    'yolo_backbone': '#4169E1',  # Blue
    'hra_fusion': '#32CD32',     # Green
    'gd_mse': '#FF8C00',         # Orange
    'hd_dsah': '#DC143C',        # Red
}

def draw_rounded_box(ax, x, y, width, height, color, text, fontsize=9, text_color='white'):
    """Draw a rounded rectangle with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=text_color, fontweight='bold', wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0, 0.3), color='black'):
    """Draw an arrow with optional label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=15,
                           color=color, linewidth=1.5)
    ax.add_patch(arrow)
    if label:
        ax.text((x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1], 
                label, ha='center', va='center', fontsize=8, 
                color='black', style='italic')

def draw_curved_arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0, 0.2), color='black'):
    """Draw a curved arrow"""
    style = "arc3,rad=0.2"
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=15,
                           connectionstyle=style,
                           color=color, linewidth=1.5)
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + 0.3
        ax.text(mid_x + label_offset[0], mid_y + label_offset[1], 
                label, ha='center', va='center', fontsize=8, 
                color='black', style='italic')

# Title
ax.text(6, 7.5, '图1 本文方法整体架构', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='black')

# ========== Layer 1: Input ==========
draw_rounded_box(ax, 1.2, 5.5, 1.8, 0.8, colors['input_output'], 
                 '输入图像\n640×640×3', fontsize=9)
draw_arrow(ax, 2.1, 5.5, 3.0, 5.5, None)

# ========== Layer 2: YOLOv11 Backbone ==========
draw_rounded_box(ax, 3.8, 5.5, 2.0, 1.0, colors['yolo_backbone'], 
                 'YOLOv11主干网络', fontsize=10)

# Arrows from backbone to F3, F4, F5
draw_arrow(ax, 4.8, 5.0, 4.8, 4.2, 'F5\n20×20×512', (0.5, 0))
draw_arrow(ax, 4.3, 5.0, 3.5, 4.2, 'F4\n40×40×256', (-0.3, 0))
draw_arrow(ax, 3.8, 5.0, 2.2, 4.2, 'F3\n80×80×128', (-0.5, 0))

# F3, F4, F5 boxes
draw_rounded_box(ax, 2.2, 3.6, 1.3, 0.7, colors['yolo_backbone'], 
                 'F3\n80×80×128', fontsize=8)
draw_rounded_box(ax, 3.5, 3.6, 1.3, 0.7, colors['yolo_backbone'], 
                 'F4\n40×40×256', fontsize=8)
draw_rounded_box(ax, 4.8, 3.6, 1.3, 0.7, colors['yolo_backbone'], 
                 'F5\n20×20×512', fontsize=8)

# Additional F2 from backbone (curved)
draw_curved_arrow(ax, 3.3, 5.0, 1.0, 3.9, 'F2\n160×160×64', (-0.4, 0.2))
draw_rounded_box(ax, 1.0, 3.6, 1.3, 0.7, colors['yolo_backbone'], 
                 'F2\n160×160×64', fontsize=8)

# ========== Layer 3: HRA-Fusion ==========
# Arrows to HRA-Fusion
draw_arrow(ax, 1.0, 3.25, 2.2, 2.5, None)
draw_arrow(ax, 2.2, 3.25, 2.6, 2.5, None)
draw_arrow(ax, 3.5, 3.25, 3.4, 2.5, None)
draw_arrow(ax, 4.8, 3.25, 4.2, 2.5, None)

draw_rounded_box(ax, 3.2, 2.2, 2.4, 0.9, colors['hra_fusion'], 
                 'HRA-Fusion模块\n(添加F2特征)', fontsize=10)

# Output arrows from HRA-Fusion
draw_arrow(ax, 2.6, 1.75, 1.8, 1.1, 'F2\n160×160×64', (-0.4, 0))
draw_arrow(ax, 3.0, 1.75, 3.0, 1.1, 'F3\n80×80×128', (0.4, 0))
draw_arrow(ax, 3.4, 1.75, 4.2, 1.1, 'F4\n40×40×256', (0.4, 0))
draw_arrow(ax, 3.8, 1.75, 5.4, 1.1, 'F5\n20×20×512', (0.5, 0))

# ========== Layer 4: GD-MSE ==========
draw_rounded_box(ax, 3.6, 0.7, 2.2, 0.9, colors['gd_mse'], 
                 'GD-MSE模块\n(增强特征)', fontsize=10)

# ========== Layer 5: HD-DSAH ==========
draw_arrow(ax, 4.7, 0.7, 6.5, 0.7, '增强特征', (0, 0.25))
draw_rounded_box(ax, 7.3, 0.7, 2.0, 0.9, colors['hd_dsah'], 
                 'HD-DSAH模块\n(动态检测头)', fontsize=10)

# ========== Layer 6: Output ==========
draw_arrow(ax, 8.3, 0.7, 9.5, 0.7, None)
draw_rounded_box(ax, 10.3, 0.7, 1.8, 1.0, colors['input_output'], 
                 '检测结果\n(类别,坐标,置信度)', fontsize=9)

# ========== Legend ==========
legend_x = 7.0
legend_y = 5.8
legend_items = [
    (colors['input_output'], '输入/输出'),
    (colors['yolo_backbone'], 'YOLOv11主干'),
    (colors['hra_fusion'], 'HRA-Fusion (创新)'),
    (colors['gd_mse'], 'GD-MSE (创新)'),
    (colors['hd_dsah'], 'HD-DSAH (创新)'),
]

# Legend box
legend_box = FancyBboxPatch((legend_x - 0.3, legend_y - 2.2), 4.0, 2.6,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='white', edgecolor='gray', linewidth=1)
ax.add_patch(legend_box)
ax.text(legend_x + 1.7, legend_y + 0.2, '图例说明', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='black')

for i, (color, label) in enumerate(legend_items):
    y_pos = legend_y - 0.4 - i * 0.45
    rect = plt.Rectangle((legend_x, y_pos - 0.12), 0.35, 0.24, 
                         facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(legend_x + 0.55, y_pos, label, ha='left', va='center', 
            fontsize=9, color='black')

# Add innovation markers
ax.text(3.2, 2.2, '★', ha='center', va='center', fontsize=14, color='gold')
ax.text(3.6, 0.7, '★', ha='center', va='center', fontsize=14, color='gold')
ax.text(7.3, 0.7, '★', ha='center', va='center', fontsize=14, color='gold')

plt.tight_layout()

# Use raw strings for Windows paths
output_dir = r'D:\jglw\yolov11-manhole-detection\paper\figures'
output_png = os.path.join(output_dir, 'fig1_architecture.png')
output_pdf = os.path.join(output_dir, 'fig1_architecture.pdf')

plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Figure saved successfully!")
print(f"PNG: {output_png}")
print(f"PDF: {output_pdf}")
