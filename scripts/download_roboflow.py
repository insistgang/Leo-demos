#!/usr/bin/env python3
"""
Roboflow数据集下载脚本
支持从Roboflow Universe下载井盖检测数据集
"""

import os
import sys
import zipfile
import requests
from pathlib import Path


# 可用的Roboflow数据集
DATASETS = {
    "sideseeing": {
        "name": "SideSeeing Manhole Dataset",
        "url": "https://universe.roboflow.com/sideseeing/manhole-cover-dataset-yolo-62sri",
        "workspace": "sideseeing",
        "project": "manhole-cover-dataset-yolo-62sri",
        "images": 1427,
        "classes": ["Broken", "Loose", "Uncovered", "Good"],
        "description": "4类井盖状态，有预训练模型"
    },
    "manhole-5k": {
        "name": "Manhole Cover Dataset 5K",
        "url": "https://universe.roboflow.com/create-dataset-for-yolo/manhole-cover-dataset-5k-images",
        "workspace": "create-dataset-for-yolo",
        "project": "manhole-cover-dataset-5k-images",
        "images": 5000,
        "classes": "多类",
        "description": "5000张图像，大规模数据集"
    },
    "road-damage": {
        "name": "Road Damage Manhole Sewer Covers",
        "url": "https://universe.roboflow.com/hazels-space/road-damage-manhole-sewers-covers/dataset/8",
        "workspace": "hazels-space",
        "project": "road-damage-manhole-sewers-covers",
        "images": 990,
        "classes": "多类",
        "description": "2024年8月更新，最新数据集"
    }
}


def print_usage():
    """打印使用说明"""
    print("=" * 60)
    print("Roboflow井盖数据集下载工具")
    print("=" * 60)
    print("\n可用数据集:")
    for key, info in DATASETS.items():
        print(f"\n  [{key}] {info['name']}")
        print(f"       图像数: {info['images']}")
        print(f"       类别: {info['classes']}")
        print(f"       描述: {info['description']}")
        print(f"       链接: {info['url']}")

    print("\n\n使用方法:")
    print("  方式1 (使用roboflow包):")
    print("    pip install roboflow")
    print("    python scripts/download_roboflow.py --dataset sideseeing --api-key YOUR_KEY")
    print("\n  方式2 (手动下载):")
    print("    访问数据集页面 -> 点击Download -> 选择YOLOv8格式")
    print("    下载后解压到 data/raw/ 目录")
    print("\n  方式3 (查看数据集详情):")
    print("    python scripts/download_roboflow.py --info sideseeing")


def print_dataset_info(dataset_key):
    """打印数据集详细信息"""
    if dataset_key not in DATASETS:
        print(f"错误: 数据集 '{dataset_key}' 不存在")
        print(f"可用数据集: {', '.join(DATASETS.keys())}")
        return

    info = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"数据集: {info['name']}")
    print(f"{'='*60}")
    print(f"图像数量: {info['images']}")
    print(f"类别: {info['classes']}")
    print(f"描述: {info['description']}")
    print(f"\nRoboflow页面:")
    print(f"  {info['url']}")
    print(f"\n手动下载步骤:")
    print(f"  1. 访问上述链接")
    print(f"  2. 点击 'Download' 按钮")
    print(f"  3. 选择 'YOLOv8' 格式")
    print(f"  4. 下载并解压到 data/raw/{dataset_key}/")
    print(f"{'='*60}\n")


def download_with_roboflow(dataset_key, api_key):
    """使用roboflow包下载数据集"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("错误: 未安装roboflow包")
        print("请运行: pip install roboflow")
        return False

    if dataset_key not in DATASETS:
        print(f"错误: 数据集 '{dataset_key}' 不存在")
        return False

    info = DATASETS[dataset_key]

    print(f"\n正在下载数据集: {info['name']}")
    print(f"Workspace: {info['workspace']}")
    print(f"Project: {info['project']}")

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(info['workspace']).project(info['project'])
        dataset = project.version(1).download("yolov8")

        print(f"\n✅ 下载完成!")
        print(f"下载位置: {dataset.location}")
        return True

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("  1. API Key无效")
        print("  2. 数据集名称或workspace名称有误")
        print("  3. 网络连接问题")
        print("\n建议使用手动下载方式")
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1]

    if command == "--help" or command == "-h":
        print_usage()
    elif command == "--info":
        if len(sys.argv) < 3:
            print("请指定数据集名称，例如: --info sideseeing")
            return
        print_dataset_info(sys.argv[2])
    elif command == "--list":
        for key, info in DATASETS.items():
            print(f"[{key}] {info['name']} - {info['images']}张")
    elif command == "--dataset":
        if len(sys.argv) < 3:
            print("请指定数据集名称")
            return
        dataset_key = sys.argv[2]

        # 检查是否有API key
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if "--api-key" in sys.argv:
            api_key_idx = sys.argv.index("--api-key")
            if api_key_idx + 1 < len(sys.argv):
                api_key = sys.argv[api_key_idx + 1]

        if not api_key:
            print("未找到API Key，请使用以下方式之一:")
            print("  1. 设置环境变量: export ROBOFLOW_API_KEY=your_key")
            print("  2. 使用命令行参数: --api-key your_key")
            print("\n或者使用手动下载方式:")
            print_dataset_info(dataset_key)
            return

        download_with_roboflow(dataset_key, api_key)
    else:
        print_usage()


if __name__ == "__main__":
    main()
