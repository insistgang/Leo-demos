import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# 定义颜色
color_input = '#E8E8E8'
color_cls = '#4A90D9'  # 蓝色 - 分类分支
color_reg = '#5CB85C'  # 绿色 - 回归分支
color_level = '#F0AD4E'  # 橙色 - 层次
color_loss = '#D9534F'  # 红色 - 损失

def draw_box(ax, x, y, width, height, text, color, fontsize=8, text_color='white', alpha=1.0):
    """绘制带文字的矩形框"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=text_color, fontweight='bold', wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=15,
                           color=color, linewidth=1.5)
    ax.add_patch(arrow)

# ==================== 输入特征 ====================
draw_box(ax, 1.5, 4, 1.8, 1.2, '输入特征\n(Input Feature)', color_input, fontsize=9, text_color='black')

# ==================== 解耦设计 ====================
draw_box(ax, 4, 6.5, 2.5, 0.8, '解耦设计 (Decoupled Head)', '#F5F5F5', fontsize=10, text_color='black')

# ==================== 分类分支 (蓝色) ====================
# 主标题
draw_box(ax, 3.5, 5, 2.2, 0.6, '分类分支', color_cls, fontsize=10)

# Conv层
draw_box(ax, 3.5, 4, 1.5, 0.6, 'Conv', color_cls, fontsize=9)

# 三级层次化分类
draw_box(ax, 3.5, 2.8, 2.5, 0.6, '三级层次化分类', color_cls, fontsize=9)

# Level 1
draw_box(ax, 2.2, 1.6, 1.6, 0.7, 'Level 1\n有/无井盖', color_level, fontsize=8, text_color='black')

# Level 2
draw_box(ax, 4.0, 1.6, 1.6, 0.7, 'Level 2\n完好/破损/缺失', color_level, fontsize=8, text_color='black')

# Level 3
draw_box(ax, 5.8, 1.6, 1.6, 0.7, 'Level 3\n轻度/中度/重度\n移位/遮挡', color_level, fontsize=7, text_color='black')

# 类别概率输出
draw_box(ax, 3.5, 0.4, 2.2, 0.6, '类别概率\nP(cls)', color_cls, fontsize=9)

# ==================== 回归分支 (绿色) ====================
# 主标题
draw_box(ax, 7.5, 5, 2.2, 0.6, '回归分支', color_reg, fontsize=10)

# Conv层
draw_box(ax, 7.5, 4, 1.5, 0.6, 'Conv', color_reg, fontsize=9)

# 边界框回归
draw_box(ax, 7.5, 2.8, 2.2, 0.6, '边界框回归', color_reg, fontsize=9)

# 输出坐标
draw_box(ax, 7.5, 1.6, 2.0, 0.8, '(x, y, w, h)\n边界框坐标', color_reg, fontsize=9)

# 回归结果输出
draw_box(ax, 7.5, 0.4, 2.2, 0.6, '检测框输出', color_reg, fontsize=9)

# ==================== 语义对齐损失 ====================
draw_box(ax, 10.5, 4, 2.2, 1.0, '语义对齐损失\n(Semantic Alignment)', color_loss, fontsize=9)

# KL散度
draw_box(ax, 10.5, 2.5, 1.8, 0.6, 'KL散度', color_loss, fontsize=9)

# MSE
draw_box(ax, 10.5, 1.5, 1.8, 0.6, 'MSE', color_loss, fontsize=9)

# ==================== 绘制连接箭头 ====================
# 输入到两个分支
arrow_color = '#666666'
draw_arrow(ax, 2.4, 4, 2.8, 4.3, arrow_color)  # 到分类
draw_arrow(ax, 2.4, 4, 6.7, 4.3, arrow_color)  # 到回归

# 分类分支内部连接
draw_arrow(ax, 3.5, 3.7, 3.5, 3.1, color_cls)
draw_arrow(ax, 3.5, 2.5, 2.2, 2.0, color_cls)  # 到Level1
draw_arrow(ax, 3.5, 2.5, 4.0, 2.0, color_cls)  # 到Level2
draw_arrow(ax, 3.5, 2.5, 5.0, 2.0, color_cls)  # 到Level3

# Level到输出
draw_arrow(ax, 2.2, 1.2, 2.8, 0.7, color_cls)
draw_arrow(ax, 4.0, 1.2, 3.5, 0.7, color_cls)
draw_arrow(ax, 5.8, 1.2, 4.2, 0.7, color_cls)

# 回归分支内部连接
draw_arrow(ax, 7.5, 3.7, 7.5, 3.1, color_reg)
draw_arrow(ax, 7.5, 2.5, 7.5, 2.0, color_reg)
draw_arrow(ax, 7.5, 1.2, 7.5, 0.7, color_reg)

# 连接到损失函数
draw_arrow(ax, 4.6, 0.4, 9.6, 2.0, color_loss)  # 分类到损失
draw_arrow(ax, 8.6, 0.4, 9.6, 1.2, color_loss)  # 回归到损失

# ==================== 层次化概率公式 ====================
formula_text = r'$P(c_{ijk}) = P(L_1=i) \cdot P(L_2=j|L_1=i) \cdot P(L_3=k|L_2=j)$'
ax.text(6, 7.2, formula_text, ha='center', va='center', fontsize=11, 
        color='#333333', style='italic')

# ==================== 图例 ====================
legend_elements = [
    mpatches.Patch(facecolor=color_cls, edgecolor='black', label='分类分支'),
    mpatches.Patch(facecolor=color_reg, edgecolor='black', label='回归分支'),
    mpatches.Patch(facecolor=color_level, edgecolor='black', label='层次化级别'),
    mpatches.Patch(facecolor=color_loss, edgecolor='black', label='损失函数')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
          framealpha=0.9, edgecolor='gray')

# ==================== 标题 ====================
ax.text(6, 7.7, '图4  HD-DSAH检测头结构', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='black')

# ==================== 注释说明 ====================
notes = [
    '注：Level1判断是否存在井盖；Level2判断井盖状态；Level3判断破损程度或特殊状态',
    '      三级分类通过条件概率相乘得到最终类别概率'
]
for i, note in enumerate(notes):
    ax.text(0.3, 0.15 - i*0.25, note, ha='left', va='center', 
            fontsize=7, color='#666666', style='italic')

plt.tight_layout()
output_path_png = r'D:\jglw\yolov11-manhole-detection\paper\figures\fig4_hd_dsah.png'
output_path_pdf = r'D:\jglw\yolov11-manhole-detection\paper\figures\fig4_hd_dsah.pdf'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()
print(f"图4已保存至: {output_path_png}")
