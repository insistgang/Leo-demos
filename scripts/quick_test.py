"""
YOLOv11创新模块快速验证脚本

快速验证所有模块的基本功能

论文: 基于多尺度特征融合与注意力机制的井盖状态智能识别方法
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_all_modules():
    """快速测试所有模块"""
    print("="*60)
    print("YOLOv11 创新模块快速验证")
    print("="*60)

    results = {}

    # 测试HRA-Fusion
    try:
        from modules.hra_fusion import HRAFusion
        hra = HRAFusion(256, 256)
        x = torch.randn(1, 256, 20, 20)
        y = hra(x)
        results["HRA-Fusion"] = f"PASSED ({x.shape} -> {y.shape})"
        print(f"  HRA-Fusion: {results['HRA-Fusion']}")
    except Exception as e:
        results["HRA-Fusion"] = f"FAILED: {e}"
        print(f"  HRA-Fusion: {results['HRA-Fusion']}")

    # 测试GD-MSE
    try:
        from modules.gd_mse import GDMSE
        gd = GDMSE(256, 4)
        features = (torch.randn(1, 256, 20, 20) for _ in range(4))
        out = gd(tuple(features))
        results["GD-MSE"] = f"PASSED (4 scales)"
        print(f"  GD-MSE: {results['GD-MSE']}")
    except Exception as e:
        results["GD-MSE"] = f"FAILED: {e}"
        print(f"  GD-MSE: {results['GD-MSE']}")

    # 测试HD-DSAH
    try:
        from modules.hd_dsah import HDDSAH
        hd = HDDSAH(256, 7)
        x = torch.randn(1, 256, 20, 20)
        out = hd(x)
        results["HD-DSAH"] = f"PASSED (cls:{out['cls'].shape}, reg:{out['reg'].shape})"
        print(f"  HD-DSAH: {results['HD-DSAH']}")
    except Exception as e:
        results["HD-DSAH"] = f"FAILED: {e}"
        print(f"  HD-DSAH: {results['HD-DSAH']}")

    # 测试模型工厂
    try:
        from modules.model import ModelFactory
        results["ModelFactory"] = "PASSED"
        print(f"  ModelFactory: {results['ModelFactory']}")
    except Exception as e:
        results["ModelFactory"] = f"FAILED: {e}"
        print(f"  ModelFactory: {results['ModelFactory']}")

    # 测试兼容模块
    try:
        from modules.model import C3k2GD, SPPFGD, DetectHD
        c3k2 = C3k2GD(64, 64)
        sppf = SPPFGD(256, 256)
        detect = DetectHD(nc=7, ch=(256, 256, 256))
        results["Compatible Modules"] = "PASSED (C3k2GD, SPPFGD, DetectHD)"
        print(f"  Compatible Modules: {results['Compatible Modules']}")
    except Exception as e:
        results["Compatible Modules"] = f"FAILED: {e}"
        print(f"  Compatible Modules: {results['Compatible Modules']}")

    # 汇总
    print("\n" + "="*60)
    print("验证结果汇总")
    print("="*60)
    passed = sum(1 for v in results.values() if "PASSED" in v)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("\n所有模块验证通过!")
        return 0
    else:
        print(f"\n{total - passed} 个模块验证失败")
        return 1

if __name__ == "__main__":
    sys.exit(test_all_modules())
