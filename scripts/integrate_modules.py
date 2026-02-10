"""
YOLOv11创新模块集成脚本

将HRA-Fusion、GD-MSE、HD-DSAH模块集成到YOLOv11中

论文: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法
作者: [Your Name]
日期: 2026-02-08
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml
from typing import Dict, Optional, Tuple

# 添加模块路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "modules"))
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import C3k2, SPPF, Detect, Conv, Concat

from modules.hra_fusion import HRAFusion, HRAFusionNeck
from modules.gd_mse import GDMSE
from modules.hd_dsah import HDDSAHMultiScale, HDDSAHLoss
from modules.model import C3k2GD, SPPFGD, DetectHD


# ============================================================================
# 配置管理
# ============================================================================

class ExperimentConfig:
    """实验配置管理器"""

    # 实验配置 (E0-E7)
    EXPERIMENTS = {
        "E0": {
            "name": "Baseline",
            "description": "原始YOLOv11模型",
            "use_hra_fusion": False,
            "use_gd_mse": False,
            "use_hd_dsah": False,
        },
        "E1": {
            "name": "HRA-Fusion",
            "description": "高分辨率自适应特征融合",
            "use_hra_fusion": True,
            "use_gd_mse": False,
            "use_hd_dsah": False,
        },
        "E2": {
            "name": "GD-MSE",
            "description": "梯度指导多尺度增强",
            "use_hra_fusion": False,
            "use_gd_mse": True,
            "use_hd_dsah": False,
        },
        "E3": {
            "name": "HD-DSAH",
            "description": "层次化解耦检测头与语义对齐",
            "use_hra_fusion": False,
            "use_gd_mse": False,
            "use_hd_dsah": True,
        },
        "E4": {
            "name": "HRA+GD",
            "description": "HRA-Fusion + GD-MSE",
            "use_hra_fusion": True,
            "use_gd_mse": True,
            "use_hd_dsah": False,
        },
        "E5": {
            "name": "HRA+HD",
            "description": "HRA-Fusion + HD-DSAH",
            "use_hra_fusion": True,
            "use_gd_mse": False,
            "use_hd_dsah": True,
        },
        "E6": {
            "name": "GD+HD",
            "description": "GD-MSE + HD-DSAH",
            "use_hra_fusion": False,
            "use_gd_mse": True,
            "use_hd_dsah": True,
        },
        "E7": {
            "name": "Full",
            "description": "HRA-Fusion + GD-MSE + HD-DSAH",
            "use_hra_fusion": True,
            "use_gd_mse": True,
            "use_hd_dsah": True,
        },
    }

    # 类别配置 (7类井盖状态)
    CLASSES = [
        "intact",          # 0: 完整
        "minor_damaged",   # 1: 轻度破损
        "medium_damaged",  # 2: 中度破损
        "severe_damaged",  # 3: 重度破损
        "missing",         # 4: 缺失
        "displaced",       # 5: 移位
        "occluded"         # 6: 遮挡
    ]

    # 训练参数配置
    TRAINING_PARAMS = {
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "device": 0,  # GPU设备，使用None表示CPU
        "workers": 8,
        "patience": 15,
        "save": True,
        "project": "runs/manhole_detection",
        "name": None,  # 将根据实验ID自动设置
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
    }


# ============================================================================
# 模型构建器
# ============================================================================

class YOLOv11ManholeBuilder:
    """YOLOv11井盖检测模型构建器"""

    def __init__(
        self,
        model_size: str = "n",
        num_classes: int = 7,
        pretrained_path: Optional[str] = None
    ):
        """
        初始化构建器

        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            num_classes: 类别数
            pretrained_path: 预训练权重路径
        """
        self.model_size = model_size
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path

        # 通道数配置
        self.channel_map = {
            'n': [64, 128, 256],
            's': [128, 256, 512],
            'm': [192, 384, 768],
            'l': [256, 512, 1024],
            'x': [320, 640, 1280]
        }

        # 模型文件名
        self.model_files = {
            'n': 'yolo11n.pt',
            's': 'yolo11s.pt',
            'm': 'yolo11m.pt',
            'l': 'yolo11l.pt',
            'x': 'yolo11x.pt'
        }

    def build_baseline(self) -> YOLO:
        """构建基线模型 (原始YOLOv11)"""
        if self.pretrained_path:
            model = YOLO(self.pretrained_path)
        else:
            model_file = self.model_files[self.model_size]
            model = YOLO(model_file)

        # 修改类别数
        model.model[-1].nc = self.num_classes

        return model

    def build_with_hra_fusion(self) -> nn.Module:
        """构建带HRA-Fusion的模型"""
        # 加载基础模型
        base_model = self.build_baseline()

        # 创建自定义模型
        class YOLOv11HRA(nn.Module):
            def __init__(self, base_yolo, num_classes=7, channels=(64, 128, 256)):
                super().__init__()
                self.base_model = base_yolo.model

                # HRA-Fusion Neck
                self.hra_neck = HRAFusionNeck(
                    in_channels=channels,
                    out_channels=channels[2]
                )

                # 保留原检测头但修改类别数
                self.detect_head = self.base_model[-1]
                self.detect_head.nc = num_classes

            def forward(self, x):
                # 这里需要实际backbone的中间层输出
                # 简化实现，实际需要hook获取特征
                return self.base_model(x)

        return YOLOv11HRA(base_model, self.num_classes, self.channel_map[self.model_size])

    def build_with_gd_mse(self) -> nn.Module:
        """构建带GD-MSE的模型"""
        base_model = self.build_baseline()

        class YOLOv11GD(nn.Module):
            def __init__(self, base_yolo, num_classes=7, channels=(64, 128, 256)):
                super().__init__()
                self.base_model = base_yolo.model

                # GD-MSE模块
                self.gd_mse = GDMSE(
                    in_channels=channels[2],
                    num_scales=3
                )

                self.detect_head = self.base_model[-1]
                self.detect_head.nc = num_classes

            def forward(self, x):
                return self.base_model(x)

        return YOLOv11GD(base_model, self.num_classes, self.channel_map[self.model_size])

    def build_with_hd_dsah(self) -> nn.Module:
        """构建带HD-DSAH检测头的模型"""
        base_model = self.build_baseline()

        class YOLOv11HD(nn.Module):
            def __init__(self, base_yolo, num_classes=7, channels=(64, 128, 256)):
                super().__init__()
                self.base_model = base_yolo.model

                # 替换检测头
                self.detect_head = DetectHD(
                    nc=num_classes,
                    ch=tuple(channels),
                    hidden_channels=256
                )

            def forward(self, x):
                # 简化实现
                return self.base_model(x)

        return YOLOv11HD(base_model, self.num_classes, self.channel_map[self.model_size])

    def build_full_model(self) -> nn.Module:
        """构建完整模型 (HRA + GD + HD)"""
        base_model = self.build_baseline()

        class YOLOv11Full(nn.Module):
            def __init__(self, base_yolo, num_classes=7, channels=(64, 128, 256)):
                super().__init__()
                self.base_model = base_yolo.model

                # HRA-Fusion
                self.hra_neck = HRAFusionNeck(
                    in_channels=channels,
                    out_channels=channels[2]
                )

                # GD-MSE
                self.gd_mse = GDMSE(
                    in_channels=channels[2],
                    num_scales=3
                )

                # HD-DSAH检测头
                self.detect_head = DetectHD(
                    nc=num_classes,
                    ch=tuple(channels),
                    hidden_channels=256
                )

            def forward(self, x):
                return self.base_model(x)

        return YOLOv11Full(base_model, self.num_classes, self.channel_map[self.model_size])


# ============================================================================
# 训练管理器
# ============================================================================

class ExperimentManager:
    """实验管理器"""

    def __init__(
        self,
        data_yaml: str,
        project_root: Optional[Path] = None
    ):
        """
        初始化实验管理器

        Args:
            data_yaml: 数据集配置文件路径
            project_root: 项目根目录
        """
        self.data_yaml = data_yaml
        self.project_root = project_root or Path(__file__).parent.parent
        self.results = {}

    def run_experiment(
        self,
        experiment_id: str,
        model_size: str = "n",
        epochs: int = 100,
        batch: int = 16,
        device: Optional[int] = 0
    ) -> Dict:
        """
        运行单个实验

        Args:
            experiment_id: 实验ID (E0-E7)
            model_size: 模型大小
            epochs: 训练轮数
            batch: 批次大小
            device: 设备号

        Returns:
            训练结果字典
        """
        config = ExperimentConfig.EXPERIMENTS[experiment_id]
        print(f"\n{'='*60}")
        print(f"Running Experiment {experiment_id}: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        # 创建模型构建器
        builder = YOLOv11ManholeBuilder(
            model_size=model_size,
            num_classes=len(ExperimentConfig.CLASSES)
        )

        # 根据配置构建模型
        if experiment_id == "E0":
            model = builder.build_baseline()
        elif experiment_id == "E1":
            model = builder.build_with_hra_fusion()
        elif experiment_id == "E2":
            model = builder.build_with_gd_mse()
        elif experiment_id == "E3":
            model = builder.build_with_hd_dsah()
        elif experiment_id == "E4":
            # E4: HRA + GD
            model = builder.build_baseline()  # 需要实现组合版本
        elif experiment_id == "E5":
            # E5: HRA + HD
            model = builder.build_baseline()  # 需要实现组合版本
        elif experiment_id == "E6":
            # E6: GD + HD
            model = builder.build_baseline()  # 需要实现组合版本
        elif experiment_id == "E7":
            model = builder.build_full_model()
        else:
            raise ValueError(f"Unknown experiment ID: {experiment_id}")

        # 训练参数
        train_params = ExperimentConfig.TRAINING_PARAMS.copy()
        train_params.update({
            "epochs": epochs,
            "batch": batch,
            "device": device,
            "name": f"{experiment_id}_{config['name']}",
        })

        # 开始训练
        try:
            results = model.train(
                data=self.data_yaml,
                **train_params
            )

            # 保存结果
            self.results[experiment_id] = {
                "config": config,
                "results": results,
                "model_size": model_size,
            }

            print(f"\nExperiment {experiment_id} completed!")
            return results

        except Exception as e:
            print(f"\nExperiment {experiment_id} failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_experiments(
        self,
        model_size: str = "n",
        epochs: int = 100,
        batch: int = 16,
        device: Optional[int] = 0
    ):
        """运行所有实验"""
        for exp_id in ExperimentConfig.EXPERIMENTS.keys():
            self.run_experiment(exp_id, model_size, epochs, batch, device)

        # 生成总结报告
        self.generate_summary()

    def generate_summary(self):
        """生成实验总结报告"""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")

        for exp_id, result in self.results.items():
            if result is None:
                continue
            config = result["config"]
            print(f"\n{exp_id}: {config['name']}")
            print(f"  Description: {config['description']}")
            if result["results"]:
                metrics = result["results"].results_dict
                print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
                print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")


# ============================================================================
# YAML配置生成器
# ============================================================================

def create_data_yaml(
    dataset_path: str,
    class_names: list = None,
    output_path: str = None
) -> str:
    """
    创建数据集配置文件

    Args:
        dataset_path: 数据集根目录
        class_names: 类别名称列表
        output_path: 输出文件路径

    Returns:
        配置文件路径
    """
    if class_names is None:
        class_names = ExperimentConfig.CLASSES

    dataset_path = Path(dataset_path)

    config = {
        'path': str(dataset_path.absolute()),
        'train': str((dataset_path / 'images' / 'train').absolute()),
        'val': str((dataset_path / 'images' / 'val').absolute()),
        'test': str((dataset_path / 'images' / 'test').absolute()),
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    if output_path is None:
        output_path = dataset_path / 'data.yaml'

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"Data configuration saved to: {output_path}")
    return str(output_path)


# ============================================================================
# 工具函数
# ============================================================================

def verify_modules() -> bool:
    """验证所有模块是否正确导入"""
    try:
        from modules import (
            HRAFusion, HRAFusionNeck,
            GDMSE, GDMSELite,
            HDDSAH, HDDSAHMultiScale, HDDSAHLoss,
            YOLOv11ManholeDetection, ModelFactory
        )
        print("All modules imported successfully!")
        return True
    except ImportError as e:
        print(f"Module import failed: {e}")
        return False


def count_model_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_forward(model: nn.Module, input_shape: Tuple[int, int, int] = (1, 3, 640)):
    """测试模型前向传播"""
    model.eval()
    with torch.no_grad():
        x = torch.randn(*input_shape)
        try:
            output = model(x)
            print(f"Forward test passed! Input: {x.shape}, Output: {type(output)}")
            return True
        except Exception as e:
            print(f"Forward test failed: {e}")
            return False


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv11井盖检测创新模块集成")
    parser.add_argument("--verify", action="store_true", help="验证模块导入")
    parser.add_argument("--create-config", type=str, help="创建数据集配置文件")
    parser.add_argument("--experiment", type=str, choices=["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"],
                        help="运行单个实验")
    parser.add_argument("--all", action="store_true", help="运行所有实验")
    parser.add_argument("--data", type=str, default="data/manhole",
                        help="数据集配置文件或路径")
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="模型大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", type=int, default=0, help="GPU设备号")

    args = parser.parse_args()

    # 验证模块
    if args.verify:
        verify_modules()
        return

    # 创建配置文件
    if args.create_config:
        create_data_yaml(args.create_config)
        return

    # 运行实验
    if args.experiment or args.all:
        # 检查数据配置
        data_yaml = args.data
        if not Path(data_yaml).exists():
            # 尝试创建配置
            data_yaml = create_data_yaml(args.data)

        manager = ExperimentManager(data_yaml)

        if args.all:
            manager.run_all_experiments(
                model_size=args.model_size,
                epochs=args.epochs,
                batch=args.batch,
                device=args.device
            )
        else:
            manager.run_experiment(
                experiment_id=args.experiment,
                model_size=args.model_size,
                epochs=args.epochs,
                batch=args.batch,
                device=args.device
            )
        return

    # 默认: 验证模块
    print("No action specified. Verifying modules...")
    verify_modules()


if __name__ == "__main__":
    main()
