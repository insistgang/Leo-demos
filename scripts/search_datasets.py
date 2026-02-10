#!/usr/bin/env python3
"""
井盖检测数据集搜索脚本
搜索公开可用的井盖/道路缺陷检测数据集
"""

import requests
from pathlib import Path
import json

# 公开数据集列表
DATASETS = {
    "kaggle_manhole": {
        "name": "Manhole Cover Detection Dataset",
        "url": "https://www.kaggle.com/datasets/",
        "search_keywords": ["manhole", "sewer", "road", "cover"],
        "type": "Kaggle"
    },
    "roaddamage": {
        "name": "Road Damage Dataset",
        "url": "https://github.com/sekilab/RoadDamageDetector/",
        "type": "GitHub",
        "classes": ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44"],
        "contains_manhole": "D44"  # D44为井盖类
    },
    "rdd2020": {
        "name": "RDD2020: Road Damage Detection",
        "url": "https://rdd2020.ethz.ch/",
        "type": "Official",
        "countries": ["Czech", "India", "Japan"],
        "contains_manhole": True
    },
    "crackforest": {
        "name": "CrackForest",
        "url": "https://github.com/cuilimeng/CrackForest",
        "type": "GitHub",
        "focus": "road crack",
        "contains_manhole": False
    }
}

def print_search_guide():
    """打印数据集搜索指南"""
    print("=" * 60)
    print("井盖检测数据集搜索指南")
    print("=" * 60)

    print("\n【方法1】Kaggle搜索")
    print("-" * 40)
    print("关键词:")
    for keyword in DATASETS["kaggle_manhole"]["search_keywords"]:
        print(f"  - {keyword}")
    print("\nURL: https://www.kaggle.com/datasets")
    print("建议搜索: 'manhole detection', 'road infrastructure'")

    print("\n【方法2】GitHub搜索")
    print("-" * 40)
    print("搜索命令:")
    print("  site:github.com manhole detection dataset")
    print("  site:github.com sewer cover yolo")
    print("  site:github.com road defect detection")

    print("\n【方法3】学术数据集")
    print("-" * 40)
    datasets = [
        ("RDD2020", "https://rdd2020.ethz.ch/"),
        ("Road Damage Dataset", "https://github.com/sekilab/RoadDamageDetector/"),
        ("CDRD2018", "https://github.com/lehaifeng/CRACK-detection")
    ]
    for name, url in datasets:
        print(f"  {name}: {url}")

    print("\n【方法4】国内平台")
    print("-" * 40)
    print("  - 天池大赛: https://tianchi.aliyun.com/")
    print("  - 飞桨: https://aistudio.baidu.com/")
    print("  - 数据堂: https://www.datatang.com/")

    print("\n" + "=" * 60)

def create_data_collection_template():
    """创建数据收集模板"""
    template = {
        "dataset_name": "",
        "source": "",
        "download_url": "",
        "total_images": 0,
        "resolution": "",
        "format": "",
        "has_annotations": False,
        "annotation_format": "",
        "classes": [],
        "license": "",
        "notes": ""
    }

    output_path = Path("data/dataset_candidates.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([template], f, indent=2, ensure_ascii=False)

    print(f"\n数据集记录模板已创建: {output_path}")
    print("请在找到数据集后填写此文件")

def main():
    print_search_guide()
    create_data_collection_template()

    print("\n下一步操作:")
    print("1. 访问上述网站搜索数据集")
    print("2. 记录找到的数据集信息到 data/dataset_candidates.json")
    print("3. 下载数据集到 data/raw/ 目录")
    print("4. 运行 python scripts/prepare_data.py 进行数据预处理")

if __name__ == '__main__':
    main()
