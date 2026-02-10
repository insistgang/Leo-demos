import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Define colors
color_input = "#E8F4FD"      # Light blue for inputs
color_gradient = "#FFF2CC"   # Light yellow for gradient extraction
color_weight = "#E1D5E7"     # Light purple for weight calculation
color_output = "#D5E8D4"     # Light green for output
color_arrow = "#666666"      # Gray for arrows

# Helper function to draw arrow
def draw_arrow(ax, start, end, color=color_arrow):
    arrow = FancyArrowPatch(start, end,
                           arrowstyle="->",
                           mutation_scale=15,
                           color=color,
                           linewidth=1.5)
    ax.add_patch(arrow)

# ==================== Left side: Multi-scale inputs ====================
# Input F3
ax.add_patch(FancyBboxPatch((0.3, 4.2), 1.2, 0.8,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor=color_input, edgecolor="#1976D2", linewidth=2))
ax.text(0.9, 4.6, r"$F_3$", ha="center", va="center", fontsize=12, weight="bold", color="#1976D2")
ax.text(0.9, 4.4, "(large scale)", ha="center", va="center", fontsize=8, color="#666666")

# Input F4
ax.add_patch(FancyBboxPatch((0.3, 2.6), 1.2, 0.8,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor=color_input, edgecolor="#1976D2", linewidth=2))
ax.text(0.9, 3.0, r"$F_4$", ha="center", va="center", fontsize=12, weight="bold", color="#1976D2")
ax.text(0.9, 2.8, "(medium scale)", ha="center", va="center", fontsize=8, color="#666666")

# Input F5
ax.add_patch(FancyBboxPatch((0.3, 1.0), 1.2, 0.8,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor=color_input, edgecolor="#1976D2", linewidth=2))
ax.text(0.9, 1.4, r"$F_5$", ha="center", va="center", fontsize=12, weight="bold", color="#1976D2")
ax.text(0.9, 1.2, "(small scale)", ha="center", va="center", fontsize=8, color="#666666")

# Label for multi-scale input
ax.text(0.9, 5.3, "Multi-scale Input", ha="center", va="center", fontsize=10, weight="bold", color="#333333")

# ==================== Middle: Gradient Information Extraction ====================
# Gradient extraction boxes
for i, (y_pos, label) in enumerate([(4.3, "Var"), (2.7, "Var"), (1.1, "Var")]):
    ax.add_patch(FancyBboxPatch((2.5, y_pos), 1.0, 0.6,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=color_gradient, edgecolor="#F57C00", linewidth=1.5))
    ax.text(3.0, y_pos + 0.3, label, ha="center", va="center", fontsize=10, weight="bold", color="#E65100")

# Arrows from inputs to gradient extraction
draw_arrow(ax, (1.5, 4.6), (2.5, 4.6))
draw_arrow(ax, (1.5, 3.0), (2.5, 3.0))
draw_arrow(ax, (1.5, 1.4), (2.5, 1.4))

# Label for gradient extraction
ax.text(3.0, 5.3, "Gradient Extraction", ha="center", va="center", fontsize=10, weight="bold", color="#333333")
ax.text(3.0, 5.0, r"$G_s(F_i) = \mathrm{Var}(F_i)$", ha="center", va="center", fontsize=9, color="#E65100")

# ==================== Right side: Gradient Sensitivity Gs ====================
# Gs boxes
for i, y_pos in enumerate([4.3, 2.7, 1.1]):
    ax.add_patch(FancyBboxPatch((4.0, y_pos), 0.9, 0.6,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="#FFECB3", edgecolor="#FF8F00", linewidth=1.5))
    ax.text(4.45, y_pos + 0.3, r"$G_s$", ha="center", va="center", fontsize=10, weight="bold", color="#E65100")

# Arrows from Var to Gs
draw_arrow(ax, (3.5, 4.6), (4.0, 4.6))
draw_arrow(ax, (3.5, 3.0), (4.0, 3.0))
draw_arrow(ax, (3.5, 1.4), (4.0, 1.4))

# ==================== Weight Calculation ====================
# Softmax weight calculation box
ax.add_patch(FancyBboxPatch((5.3, 2.2), 1.6, 1.6,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor=color_weight, edgecolor="#7B1FA2", linewidth=2))
ax.text(6.1, 3.4, "Weight Calculation", ha="center", va="center", fontsize=10, weight="bold", color="#7B1FA2")
ax.text(6.1, 3.0, "Softmax", ha="center", va="center", fontsize=9, color="#7B1FA2")
ax.text(6.1, 2.6, r"$w_i = \frac{e^{G_s(F_i)}}{\sum_j e^{G_s(F_j)}}$", 
        ha="center", va="center", fontsize=8, color="#333333")

# Arrows from Gs to weight calculation (converging)
draw_arrow(ax, (4.9, 4.6), (5.8, 3.8))
draw_arrow(ax, (4.9, 3.0), (5.8, 3.0))
draw_arrow(ax, (4.9, 1.4), (5.8, 2.2))

# ==================== Weighted Aggregation ====================
# Weighted aggregation box
ax.add_patch(FancyBboxPatch((5.3, 0.5), 1.6, 1.2,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor="#C8E6C9", edgecolor="#388E3C", linewidth=2))
ax.text(6.1, 1.3, "Weighted Aggregation", ha="center", va="center", fontsize=10, weight="bold", color="#2E7D32")
ax.text(6.1, 0.9, r"$F_{agg} = \sum_i w_i \cdot F_i$", 
        ha="center", va="center", fontsize=8, color="#333333")

# Arrow from weight calculation to aggregation
draw_arrow(ax, (6.1, 2.2), (6.1, 1.7))

# ==================== Output ====================
# Output box
ax.add_patch(FancyBboxPatch((7.5, 2.4), 1.8, 1.2,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor=color_output, edgecolor="#2E7D32", linewidth=2))
ax.text(8.4, 3.2, r"$F_{agg}$", ha="center", va="center", fontsize=14, weight="bold", color="#2E7D32")
ax.text(8.4, 2.8, "(Aggregated Feature)", ha="center", va="center", fontsize=9, color="#666666")

# Arrow from aggregation to output
draw_arrow(ax, (6.9, 1.8), (7.5, 2.8))

# Arrow from weighted aggregation to output
draw_arrow(ax, (6.9, 1.1), (7.5, 2.6))

# ==================== Title ====================
ax.text(5, 5.8, "Figure 3: GD-MSE Module Structure", ha="center", va="center", fontsize=14, weight="bold", color="#333333")

# ==================== Additional annotations ====================
# Add small formula annotations near arrows
ax.text(2.0, 4.8, r"$F_3$", ha="center", va="center", fontsize=8, color="#666666")
ax.text(2.0, 3.2, r"$F_4$", ha="center", va="center", fontsize=8, color="#666666")
ax.text(2.0, 1.6, r"$F_5$", ha="center", va="center", fontsize=8, color="#666666")

# Add dimension labels
ax.text(0.9, 0.3, r"$\in \mathbb{R}^{C \times H \times W}$", ha="center", va="center", fontsize=7, color="#666666")

plt.tight_layout()
plt.savefig("D:/jglw/yolov11-manhole-detection/paper/figures/fig3_gd_mse.png", 
            dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.savefig("D:/jglw/yolov11-manhole-detection/paper/figures/fig3_gd_mse.pdf", 
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("Figure saved successfully\!")
