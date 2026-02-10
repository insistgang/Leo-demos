"""
消融实验评估脚本

用于评估创新模块（HRA-Fusion、GD-MSE、HD-DSAH）的独立贡献

实验矩阵:
    E0: YOLOv11n baseline
    E1: +HRA-Fusion
    E2: +GD-MSE
    E3: +HD-DSAH
    E4: HRA-Fusion + GD-MSE
    E5: HRA-Fusion + HD-DSAH
    E6: GD-MSE + HD-DSAH
    E7: Full (HRA-Fusion + GD-MSE + HD-DSAH)

论文: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法
作者: [Your Name]
日期: 2026-02-08
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    use_hra_fusion: bool = False
    use_gd_mse: bool = False
    use_hd_dsah: bool = False
    description: str = ""

    def get_modules(self) -> List[str]:
        """获取启用的模块列表"""
        modules = []
        if self.use_hra_fusion:
            modules.append("HRA-Fusion")
        if self.use_gd_mse:
            modules.append("GD-MSE")
        if self.use_hd_dsah:
            modules.append("HD-DSAH")
        return modules if modules else ["Baseline"]


# 定义所有实验配置
EXPERIMENTS = {
    "E0": ExperimentConfig(
        name="E0",
        use_hra_fusion=False,
        use_gd_mse=False,
        use_hd_dsah=False,
        description="YOLOv11n baseline"
    ),
    "E1": ExperimentConfig(
        name="E1",
        use_hra_fusion=True,
        use_gd_mse=False,
        use_hd_dsah=False,
        description="+HRA-Fusion (P2层 + 双分支融合)"
    ),
    "E2": ExperimentConfig(
        name="E2",
        use_hra_fusion=False,
        use_gd_mse=True,
        use_hd_dsah=False,
        description="+GD-MSE (梯度指导多尺度增强)"
    ),
    "E3": ExperimentConfig(
        name="E3",
        use_hra_fusion=False,
        use_gd_mse=False,
        use_hd_dsah=True,
        description="+HD-DSAH (层次化解耦检测头)"
    ),
    "E4": ExperimentConfig(
        name="E4",
        use_hra_fusion=True,
        use_gd_mse=True,
        use_hd_dsah=False,
        description="HRA-Fusion + GD-MSE"
    ),
    "E5": ExperimentConfig(
        name="E5",
        use_hra_fusion=True,
        use_gd_mse=False,
        use_hd_dsah=True,
        description="HRA-Fusion + HD-DSAH"
    ),
    "E6": ExperimentConfig(
        name="E6",
        use_hra_fusion=False,
        use_gd_mse=True,
        use_hd_dsah=True,
        description="GD-MSE + HD-DSAH"
    ),
    "E7": ExperimentConfig(
        name="E7",
        use_hra_fusion=True,
        use_gd_mse=True,
        use_hd_dsah=True,
        description="Full (HRA-Fusion + GD-MSE + HD-DSAH)"
    ),
}


# ============================================================================
# 评估指标
# ============================================================================

@dataclass
class Metrics:
    """评估指标"""
    # 检测精度指标
    mAP_05: float = 0.0        # mAP@0.5
    mAP_05_095: float = 0.0    # mAP@0.5:0.95
    precision: float = 0.0     # 精确率
    recall: float = 0.0        # 召回率
    f1_score: float = 0.0      # F1分数

    # 小目标检测指标
    ap_small: float = 0.0      # 小目标AP (<32²)
    ap_medium: float = 0.0     # 中目标AP (32²-96²)
    ap_large: float = 0.0      # 大目标AP (>96²)

    # 分类指标
    accuracy: float = 0.0      # 分类准确率
    per_class_ap: List[float] = None  # 各类别AP

    # 效率指标
    fps: float = 0.0           # 每秒帧数
    params: float = 0.0        # 参数量 (M)
    flops: float = 0.0         # FLOPs (G)
    inference_time: float = 0.0 # 单帧推理时间 (ms)

    def __post_init__(self):
        if self.per_class_ap is None:
            self.per_class_ap = [0.0] * 7  # 7类状态

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


# ============================================================================
# 模型构建器
# ============================================================================

class ModelBuilder:
    """模型构建器 - 根据配置构建不同模块组合的模型"""

    def __init__(self, base_model_path: Optional[str] = None):
        self.base_model_path = base_model_path

    def build(self, config: ExperimentConfig) -> nn.Module:
        """
        根据配置构建模型

        Args:
            config: 实验配置

        Returns:
            构建好的模型
        """
        # 这里需要实际的YOLOv11模型导入
        # 示例代码，实际使用时需要替换
        from ultralytics import YOLO

        # 加载基础模型
        model = YOLO("yolo11n.pt")

        # 根据配置添加模块
        if config.use_hra_fusion:
            # 添加HRA-Fusion模块
            pass  # 实际实现需要修改模型结构

        if config.use_gd_mse:
            # 添加GD-MSE模块
            pass

        if config.use_hd_dsah:
            # 替换检测头为HD-DSAH
            pass

        return model

    def get_model_info(self, model: nn.Module) -> Tuple[float, float]:
        """获取模型参数量和FLOPs"""
        # 参数量
        params = sum(p.numel() for p in model.parameters()) / 1e6

        # FLOPs (需要使用thop或其他工具)
        try:
            from thop import profile
            dummy_input = torch.randn(1, 3, 640, 640)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops = flops / 1e9
        except:
            flops = 0.0

        return params, flops


# ============================================================================
# 评估器
# ============================================================================

class Evaluator:
    """模型评估器"""

    def __init__(
        self,
        data_path: str,
        batch_size: int = 16,
        num_workers: int = 4,
        device: str = "cuda"
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def evaluate(self, model: nn.Module, experiment_id: str) -> Metrics:
        """
        评估模型

        Args:
            model: 要评估的模型
            experiment_id: 实验ID

        Returns:
            评估指标
        """
        model.eval()

        # 这里需要实际的数据加载和评估逻辑
        # 示例代码，实际使用时需要实现

        metrics = Metrics()

        # 测量推理速度
        inference_times = []
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)

        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                inference_times.append((end - start) * 1000)  # ms

        metrics.inference_time = np.mean(inference_times[10:])  # 跳过前10次warmup
        metrics.fps = 1000 / metrics.inference_time

        # 这里需要实际的评估逻辑
        # metrics.mAP_05 = ...
        # metrics.ap_small = ...
        # ...

        return metrics


# ============================================================================
# 实验管理器
# ============================================================================

class ExperimentManager:
    """消融实验管理器"""

    def __init__(
        self,
        output_dir: str = "./results/ablation",
        data_path: str = "./data",
        device: str = "cuda"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path
        self.device = device

        self.model_builder = ModelBuilder()
        self.evaluator = Evaluator(data_path, device=device)

        # 存储实验结果
        self.results: Dict[str, Dict] = {}

    def run_experiment(self, exp_id: str, config: ExperimentConfig) -> Dict:
        """
        运行单个实验

        Args:
            exp_id: 实验ID (如"E0", "E1", ...)
            config: 实验配置

        Returns:
            实验结果字典
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment {exp_id}: {config.description}")
        print(f"Modules: {', '.join(config.get_modules())}")
        print(f"{'='*60}")

        # 构建模型
        print(f"\n[1/4] Building model...")
        model = self.model_builder.build(config)
        model = model.to(self.device)

        # 获取模型信息
        params, flops = self.model_builder.get_model_info(model)

        # 评估模型
        print(f"[2/4] Evaluating model...")
        metrics = self.evaluator.evaluate(model, exp_id)
        metrics.params = params
        metrics.flops = flops

        # 保存结果
        print(f"[3/4] Saving results...")
        result = {
            "experiment_id": exp_id,
            "config": asdict(config),
            "metrics": metrics.to_dict(),
            "modules": config.get_modules()
        }

        self.results[exp_id] = result
        self._save_result(exp_id, result)

        # 打印结果摘要
        print(f"[4/4] Results:")
        print(f"  mAP@0.5: {metrics.mAP_05:.2f}%")
        print(f"  AP_small: {metrics.ap_small:.2f}%")
        print(f"  Accuracy: {metrics.accuracy:.2f}%")
        print(f"  FPS: {metrics.fps:.1f}")
        print(f"  Parameters: {params:.2f}M")

        return result

    def run_all(self, experiments: Optional[List[str]] = None) -> Dict:
        """
        运行所有实验

        Args:
            experiments: 要运行的实验ID列表，None表示运行全部

        Returns:
            所有实验结果
        """
        if experiments is None:
            experiments = list(EXPERIMENTS.keys())

        for exp_id in experiments:
            config = EXPERIMENTS[exp_id]
            self.run_experiment(exp_id, config)

        # 生成对比报告
        self._generate_report(experiments)

        return self.results

    def _save_result(self, exp_id: str, result: Dict):
        """保存单个实验结果"""
        result_file = self.output_dir / f"{exp_id}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    def _generate_report(self, experiments: List[str]):
        """生成实验对比报告"""
        # 创建对比表格
        report_lines = []
        report_lines.append("\n" + "="*100)
        report_lines.append("消融实验对比报告")
        report_lines.append("="*100)
        report_lines.append(f"{'实验ID':<8} {'配置':<40} {'mAP@0.5':<10} {'AP_S':<8} {'Acc':<8} {'FPS':<8} {'Params':<10}")
        report_lines.append("-"*100)

        for exp_id in experiments:
            result = self.results.get(exp_id)
            if result is None:
                continue

            metrics = result["metrics"]
            config = result["config"]
            modules = result["modules"]

            config_str = ", ".join(modules) if modules else "Baseline"
            report_lines.append(
                f"{exp_id:<8} {config_str:<40} "
                f"{metrics['mAP_05']:<10.2f} "
                f"{metrics['ap_small']:<8.2f} "
                f"{metrics['accuracy']:<8.2f} "
                f"{metrics['fps']:<8.1f} "
                f"{metrics['params']:<10.2f}"
            )

        report_lines.append("="*100)

        # 计算模块贡献
        report_lines.append("\n模块贡献分析:")
        baseline = self.results.get("E0")
        if baseline:
            baseline_map = baseline["metrics"]["mAP_05"]
            for exp_id in ["E1", "E2", "E3"]:
                result = self.results.get(exp_id)
                if result:
                    metrics = result["metrics"]
                    modules = result["modules"]
                    improvement = metrics["mAP_05"] - baseline_map
                    report_lines.append(
                        f"  {modules[0] if modules else 'Unknown'}: "
                        f"+{improvement:.2f}% mAP@0.5"
                    )

        # 打印报告
        report = "\n".join(report_lines)
        print(report)

        # 保存报告
        report_file = self.output_dir / "ablation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        # 生成Markdown格式的报告（用于论文）
        self._generate_markdown_report(experiments)

    def _generate_markdown_report(self, experiments: List[str]):
        """生成Markdown格式的报告"""
        md_lines = []
        md_lines.append("# 消融实验结果")
        md_lines.append("\n## 实验配置")
        md_lines.append("\n| 实验ID | HRA-Fusion | GD-MSE | HD-DSAH | 描述 |")
        md_lines.append("|--------|------------|--------|---------|------|")

        for exp_id in experiments:
            config = EXPERIMENTS[exp_id]
            md_lines.append(
                f"| {exp_id} | {'✓' if config.use_hra_fusion else '✗'} | "
                f"{'✓' if config.use_gd_mse else '✗'} | "
                f"{'✓' if config.use_hd_dsah else '✗'} | "
                f"{config.description} |"
            )

        md_lines.append("\n## 性能对比")
        md_lines.append("\n| 实验ID | mAP@0.5 (%) | AP_S (%) | Acc (%) | FPS | Params (M) |")
        md_lines.append("|--------|-------------|---------|---------|-----|------------|")

        for exp_id in experiments:
            result = self.results.get(exp_id)
            if result:
                m = result["metrics"]
                md_lines.append(
                    f"| {exp_id} | {m['mAP_05']:.2f} | {m['ap_small']:.2f} | "
                    f"{m['accuracy']:.2f} | {m['fps']:.1f} | {m['params']:.2f} |"
                )

        md_lines.append("\n## 模块贡献分析")
        md_lines.append("\n基于E0 baseline，各模块的贡献：")
        md_lines.append("\n| 模块 | mAP@0.5提升 | AP_S提升 |")
        md_lines.append("|------|-------------|----------|")

        baseline = self.results.get("E0")
        if baseline:
            baseline_map = baseline["metrics"]["mAP_05"]
            baseline_aps = baseline["metrics"]["ap_small"]

            for exp_id, module_name in [("E1", "HRA-Fusion"), ("E2", "GD-MSE"), ("E3", "HD-DSAH")]:
                result = self.results.get(exp_id)
                if result:
                    m = result["metrics"]
                    map_improve = m["mAP_05"] - baseline_map
                    aps_improve = m["ap_small"] - baseline_aps
                    md_lines.append(f"| {module_name} | +{map_improve:.2f}% | +{aps_improve:.2f}% |")

        # 保存Markdown报告
        md_file = self.output_dir / "ablation_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))

        print(f"\nMarkdown report saved to: {md_file}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    import argparse

    parser = argparse.ArgumentParser(description="消融实验评估脚本")
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=["E0", "E1", "E2", "E3", "E7"],
                        help="要运行的实验ID列表")
    parser.add_argument("--data", type=str, default="./data",
                        help="数据集路径")
    parser.add_argument("--output", type=str, default="./results/ablation",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")

    args = parser.parse_args()

    # 创建实验管理器
    manager = ExperimentManager(
        output_dir=args.output,
        data_path=args.data,
        device=args.device
    )

    # 运行实验
    print("开始运行消融实验...")
    print(f"实验列表: {', '.join(args.experiments)}")
    print(f"数据路径: {args.data}")
    print(f"输出目录: {args.output}")
    print(f"设备: {args.device}")

    results = manager.run_all(args.experiments)

    print("\n所有实验完成!")
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
