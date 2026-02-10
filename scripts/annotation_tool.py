#!/usr/bin/env python3
"""
井盖检测数据标注工具
支持图像标注、YOLO格式导出、数据集管理
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, Canvas, Frame, Button, Label, Entry, messagebox
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw
import yaml

class ManholeAnnotationTool:
    """井盖标注工具"""

    # 井盖类别定义
    CLASSES = {
        0: "intact",           # 完整
        1: "minor_damaged",    # 轻度破损
        2: "medium_damaged",   # 中度破损
        3: "severe_damaged",   # 重度破损
        4: "missing",          # 缺失
        5: "displaced",        # 移位
        6: "occluded"          # 遮挡
    }

    # 颜色定义
    COLORS = {
        0: (0, 255, 0),      # 绿色 - 完整
        1: (255, 255, 0),    # 黄色 - 轻度破损
        2: (255, 165, 0),    # 橙色 - 中度破损
        3: (0, 0, 255),      # 红色 - 重度破损
        4: (128, 0, 128),    # 紫色 - 缺失
        5: (255, 0, 255),    # 品红 - 移位
        6: (128, 128, 128)   # 灰色 - 遮挡
    }

    def __init__(self, image_dir, output_dir, class_names=None):
        """
        初始化标注工具

        Args:
            image_dir: 图像目录
            output_dir: 输出目录（标签文件）
            class_names: 自定义类别名称
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 类别名称
        self.class_names = class_names or list(self.CLASSES.values())

        # 加载图像列表
        self.image_files = self._load_images()
        self.current_index = 0

        # 当前图像的标注
        self.current_image = None
        self.current_annotations = []  # [(class_id, x_center, y_center, width, height)]
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.current_class = 0

        # GUI相关
        self.root = None
        self.canvas = None
        self.scale = 1.0
        self.img_width = 0
        self.img_height = 0

        # 快捷键对应类别
        self.hotkeys = {
            '1': 0, '2': 1, '3': 2, '4': 3,
            '5': 4, '6': 5, '7': 6, '0': 0
        }

    def _load_images(self):
        """加载图像文件列表"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        files = []
        for ext in extensions:
            files.extend(list(self.image_dir.glob(ext)))
        return sorted(files)

    def _load_existing_annotations(self, image_path):
        """加载已存在的标注"""
        label_path = self.output_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            annotations = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        annotations.append((int(class_id), x, y, w, h))
            return annotations
        return []

    def _save_annotations(self):
        """保存当前标注"""
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]
        label_path = self.output_dir / f"{image_path.stem}.txt"

        with open(label_path, 'w') as f:
            for ann in self.current_annotations:
                class_id, x, y, w, h = ann
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    def _yolo_to_pixel(self, x, y, w, h):
        """YOLO格式转像素坐标"""
        x1 = int((x - w/2) * self.img_width)
        y1 = int((y - h/2) * self.img_height)
        x2 = int((x + w/2) * self.img_width)
        y2 = int((y + h/2) * self.img_height)
        return x1, y1, x2, y2

    def _pixel_to_yolo(self, x1, y1, x2, y2):
        """像素坐标转YOLO格式"""
        x_center = ((x1 + x2) / 2) / self.img_width
        y_center = ((y1 + y2) / 2) / self.img_height
        width = (x2 - x1) / self.img_width
        height = (y2 - y1) / self.img_height
        return x_center, y_center, width, height

    def _load_image(self, index):
        """加载指定索引的图像"""
        if 0 <= index < len(self.image_files):
            self.current_index = index
            image_path = self.image_files[index]

            # 读取图像
            self.current_image = cv2.imread(str(image_path))
            if self.current_image is None:
                return False

            self.img_height, self.img_width = self.current_image.shape[:2]

            # 加载已有标注
            self.current_annotations = self._load_existing_annotations(image_path)

            return True
        return False

    def create_gui(self):
        """创建GUI界面"""
        self.root = Tk()
        self.root.title("井盖检测标注工具")
        self.root.geometry("1200x800")

        # 顶部控制栏
        control_frame = Frame(self.root)
        control_frame.pack(side='top', fill='x', padx=5, pady=5)

        # 文件名显示
        self.file_label = Label(control_frame, text="", font=("Arial", 12))
        self.file_label.pack(side='left', padx=10)

        # 类别选择
        class_frame = Frame(control_frame)
        class_frame.pack(side='left', padx=10)

        Label(class_frame, text="当前类别:").pack(side='left')
        self.class_label = Label(class_frame, text=self.class_names[0],
                                fg="green", font=("Arial", 12, "bold"))
        self.class_label.pack(side='left', padx=5)

        # 导航按钮
        nav_frame = Frame(control_frame)
        nav_frame.pack(side='left', padx=10)

        Button(nav_frame, text="上一张 (A)", command=self.prev_image,
               width=12).pack(side='left', padx=2)
        Button(nav_frame, text="下一张 (D)", command=self.next_image,
               width=12).pack(side='left', padx=2)
        Button(nav_frame, text="跳转", command=self.jump_to_image,
               width=8).pack(side='left', padx=2)

        # 操作按钮
        action_frame = Frame(control_frame)
        action_frame.pack(side='left', padx=10)

        Button(action_frame, text="删除选中 (Del)", command=self.delete_selected,
               width=15, bg="#ffcccc").pack(side='left', padx=2)
        Button(action_frame, text="清除全部", command=self.clear_all,
               width=10, bg="#ffcccc").pack(side='left', padx=2)
        Button(action_frame, text="保存 (S)", command=self.save_and_next,
               width=10, bg="#ccffcc").pack(side='left', padx=2)

        # 进度显示
        self.progress_label = Label(control_frame, text="0 / 0")
        self.progress_label.pack(side='right', padx=10)

        # 主画布
        self.canvas = Canvas(self.root, bg='gray')
        self.canvas.pack(side='top', fill='both', expand=True)

        # 绑定事件
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Button-3>', self.on_right_click)  # 右键删除

        # 绑定键盘快捷键
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Key>', self.on_key_press)
        self.root.bind('<Delete>', lambda e: self.delete_selected())

        # 底部类别栏
        self._create_class_bar()

        # 加载第一张图像
        if self.image_files:
            self._load_image(0)
            self._update_display()

        # 居中窗口
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

        self.root.mainloop()

    def _create_class_bar(self):
        """创建底部类别选择栏"""
        class_bar = Frame(self.root)
        class_bar.pack(side='bottom', fill='x', padx=5, pady=5)

        Label(class_bar, text="快捷键选择类别:").pack(side='left')

        for i, name in enumerate(self.class_names):
            btn = Button(class_bar, text=f"{i+1}.{name}",
                       command=lambda idx=i: self.set_class(idx),
                       width=15)
            btn.pack(side='left', padx=2)

        Label(class_bar, text="| 操作: 鼠标拖拽标注 | Del删除 | S保存 | A/D前后 |").pack(side='left', padx=10)

    def set_class(self, class_id):
        """设置当前标注类别"""
        self.current_class = class_id
        color = self.COLORS.get(class_id, (128, 128, 128))
        hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        self.class_label.config(text=self.class_names[class_id], fg=hex_color)

    def on_key_press(self, event):
        """键盘事件处理"""
        key = event.keysym

        # 数字键切换类别
        if key in self.hotkeys:
            self.set_class(self.hotkeys[key])
        # A键上一张
        elif key.lower() == 'a':
            self.prev_image()
        # D键下一张
        elif key.lower() == 'd':
            self.next_image()
        # S键保存
        elif key.lower() == 's':
            self.save_and_next()

    def on_mouse_down(self, event):
        """鼠标按下"""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        """鼠标拖动"""
        if self.drawing:
            self._update_display()

    def on_mouse_up(self, event):
        """鼠标释放"""
        if self.drawing:
            self.drawing = False

            # 转换为图像坐标
            x1 = min(self.start_x, event.x) / self.scale
            y1 = min(self.start_y, event.y) / self.scale
            x2 = max(self.start_x, event.x) / self.scale
            y2 = max(self.start_y, event.y) / self.scale

            # 转换为YOLO格式
            x_center, y_center, width, height = self._pixel_to_yolo(x1, y1, x2, y2)

            # 添加标注
            self.current_annotations.append((self.current_class, x_center, y_center, width, height))
            self._update_display()

    def on_right_click(self, event):
        """右键删除最近的标注框"""
        if self.current_annotations:
            # 找到最近的标注
            x, y = event.x / self.scale, event.y / self.scale

            min_dist = float('inf')
            closest_idx = -1

            for i, ann in enumerate(self.current_annotations):
                class_id, x_center, y_center, w, h = ann
                px, py = x_center * self.img_width, y_center * self.img_height
                dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5

                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            if closest_idx >= 0 and min_dist < 50:
                del self.current_annotations[closest_idx]
                self._update_display()

    def delete_selected(self):
        """删除最后一个标注"""
        if self.current_annotations:
            self.current_annotations.pop()
            self._update_display()

    def clear_all(self):
        """清除所有标注"""
        if messagebox.askyesno("确认", "确定要清除所有标注吗？"):
            self.current_annotations = []
            self._update_display()

    def prev_image(self):
        """上一张图像"""
        if self.current_index > 0:
            self._save_annotations()
            self._load_image(self.current_index - 1)
            self._update_display()

    def next_image(self):
        """下一张图像"""
        if self.current_index < len(self.image_files) - 1:
            self._save_annotations()
            self._load_image(self.current_index + 1)
            self._update_display()

    def jump_to_image(self):
        """跳转到指定图像"""
        index = simpledialog.askinteger("跳转", f"输入图像索引 (1-{len(self.image_files)}):",
                                       minvalue=1, maxvalue=len(self.image_files))
        if index:
            self._save_annotations()
            self._load_image(index - 1)
            self._update_display()

    def save_and_next(self):
        """保存并跳到下一张"""
        self._save_annotations()
        if self.current_index < len(self.image_files) - 1:
            self._load_image(self.current_index + 1)
            self._update_display()
        else:
            messagebox.showinfo("完成", "所有图像已标注完成！")

    def _update_display(self):
        """更新显示"""
        if not self.current_image:
            return

        # 更新文件名和进度
        filename = self.image_files[self.current_index].name
        self.file_label.config(text=f"文件: {filename}")
        self.progress_label.config(text=f"{self.current_index + 1} / {len(self.image_files)}")

        # 复制图像用于显示
        display_img = self.current_image.copy()

        # 绘制已有标注框
        for ann in self.current_annotations:
            class_id, x, y, w, h = ann
            x1, y1, x2, y2 = self._yolo_to_pixel(x, y, w, h)
            color = self.COLORS.get(class_id, (128, 128, 128))

            # 绘制矩形
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)

            # 绘制类别标签
            label = self.class_names[class_id]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(display_img, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制当前拖拽框
        if self.drawing:
            # 需要获取当前鼠标位置
            x = self.root.winfo_pointerx() - self.root.winfo_rootx()
            y = self.root.winfo_pointery() - self.root.winfo_rooty()

            # 检查是否在画布内
            canvas_x = x - self.canvas.winfo_rootx()
            canvas_y = y - self.canvas.winfo_rooty()

            if 0 <= canvas_x <= self.canvas.winfo_width() and 0 <= canvas_y <= self.canvas.winfo_height():
                x1 = min(self.start_x, canvas_x) / self.scale
                y1 = min(self.start_y, canvas_y) / self.scale
                x2 = max(self.start_x, canvas_x) / self.scale
                y2 = max(self.start_y, canvas_y) / self.scale

                color = self.COLORS.get(self.current_class, (128, 128, 128))
                cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 转换为RGB
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        # 计算缩放以适应窗口
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            scale_x = canvas_width / self.img_width
            scale_y = canvas_height / self.img_height
            self.scale = min(scale_x, scale_y, 1.0)  # 不放大

            new_width = int(self.img_width * self.scale)
            new_height = int(self.img_height * self.scale)

            display_img = cv2.resize(display_img, (new_width, new_height))

        # 转换为PIL图像
        pil_image = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(pil_image)

        # 更新画布
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)


def create_data_yaml(output_dir, class_names):
    """创建data.yaml配置文件"""
    config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    return yaml_path


def main():
    """主函数"""
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    # 选择图像目录
    image_dir = filedialog.askdirectory(title="选择图像目录")
    if not image_dir:
        return

    # 选择输出目录
    output_dir = filedialog.askdirectory(title="选择标签输出目录")
    if not output_dir:
        return

    root.destroy()

    # 创建标注工具
    tool = ManholeAnnotationTool(image_dir, output_dir)

    # 创建data.yaml
    create_data_yaml(Path(output_dir).parent, tool.class_names)

    # 启动GUI
    print("=" * 50)
    print("井盖检测标注工具")
    print("=" * 50)
    print(f"图像目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(tool.image_files)} 张图像")
    print("=" * 50)
    print("\n快捷键:")
    print("  1-7: 切换标注类别")
    print("  A/D: 上一张/下一张")
    print("  S: 保存并下一张")
    print("  Del: 删除最后一个标注")
    print("  鼠标左键拖拽: 绘制标注框")
    print("  鼠标右键: 删除附近标注")
    print("=" * 50)

    tool.create_gui()


if __name__ == '__main__':
    main()
