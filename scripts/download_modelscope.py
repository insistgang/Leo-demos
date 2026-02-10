#!/usr/bin/env python3
"""
ModelScope数据集下载脚本
下载阿里云ModelScope平台上的井盖检测数据集
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# ModelScope井盖数据集列表
DATASETS = {
    "manhole_basic": {
        "name": "下水井盖数据集",
        "dataset_id": "xisowei666/xyxr_datasets",
        "description": "10500张下水井盖图像，VOC+YOLO格式",
        "size": "约500MB",
        "url": "https://www.modelscope.cn/datasets/xisowei666/xyxr_datasets"
    },
    "manhole_switch": {
        "name": "下水井盖开关数据集",
        "dataset_id": "xisowei666/manhole_switch",
        "description": "20000张下水井盖开关状态图像",
        "size": "约1GB",
        "url": "https://www.modelscope.cn/datasets/xisowei666/manhole_switch"
    }
}


class ModelScopeDownloader:
    """ModelScope数据集下载器"""

    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, output_path, chunk_size=8192):
        """
        下载文件（带进度条）

        Args:
            url: 下载链接
            output_path: 输出路径
            chunk_size: 块大小
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"正在下载: {url}")
        print(f"保存到: {output_path}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                          desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"✅ 下载完成: {output_path}")
            return True

        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False

    def extract_zip(self, zip_path, extract_to=None):
        """
        解压zip文件

        Args:
            zip_path: zip文件路径
            extract_to: 解压目录，默认为zip文件同目录
        """
        zip_path = Path(zip_path)

        if extract_to is None:
            extract_to = zip_path.parent
        else:
            extract_to = Path(extract_to)

        extract_to.mkdir(parents=True, exist_ok=True)

        print(f"正在解压: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 获取文件列表
                file_list = zip_ref.namelist()

                # 解压文件（带进度条）
                for file in tqdm(file_list, desc="解压进度"):
                    zip_ref.extract(file, extract_to)

            print(f"✅ 解压完成: {extract_to}")

            # 删除zip文件（可选）
            # zip_path.unlink()
            # print(f"已删除: {zip_path}")

            return True

        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return False

    def download_modelscope_dataset(self, dataset_key, use_api=True):
        """
        下载ModelScope数据集

        Args:
            dataset_key: 数据集键名 (manhole_basic, manhole_switch)
            use_api: 是否使用ModelScope API
        """
        if dataset_key not in DATASETS:
            print(f"❌ 未知的数据集: {dataset_key}")
            print(f"可选的数据集: {list(DATASETS.keys())}")
            return False

        dataset_info = DATASETS[dataset_key]

        print("=" * 60)
        print(f"数据集名称: {dataset_info['name']}")
        print(f"数据集ID: {dataset_info['dataset_id']}")
        print(f"描述: {dataset_info['description']}")
        print(f"大小: {dataset_info['size']}")
        print("=" * 60)

        if use_api:
            return self._download_via_api(dataset_info)
        else:
            return self._download_via_browser(dataset_info)

    def _download_via_api(self, dataset_info):
        """通过ModelScope API下载"""
        try:
            # 尝试安装modelscope
            import importlib
            if importlib.util.find_spec("modelscope") is None:
                print("正在安装ModelScope...")
                os.system("pip install modelscope -q")

            from modelscope.msdatasets import MsDataset

            print(f"\n使用ModelScope API下载数据集...")
            print(f"数据集ID: {dataset_info['dataset_id']}")

            # 下载数据集
            dataset = MsDataset.load(
                dataset_info['dataset_id'],
                split='train',
                cache_dir=self.output_dir
            )

            print(f"✅ 数据集已下载到: {self.output_dir}")
            return True

        except ImportError:
            print("❌ ModelScope安装失败")
            return self._download_via_browser(dataset_info)
        except Exception as e:
            print(f"❌ API下载失败: {e}")
            print(f"\n尝试使用浏览器下载方式...")
            return self._download_via_browser(dataset_info)

    def _download_via_browser(self, dataset_info):
        """通过浏览器下载方式"""
        print("\n" + "=" * 60)
        print("浏览器下载指南")
        print("=" * 60)
        print("\n由于ModelScope需要登录验证，请按以下步骤手动下载：\n")

        print("步骤1: 访问数据集页面")
        print(f"  URL: {dataset_info['url']}\n")

        print("步骤2: 登录ModelScope账号")
        print("  如无账号，请先注册: https://www.modelscope.cn/user/register\n")

        print("步骤3: 点击"下载全部"或下载特定文件\n")

        print("步骤4: 将下载的文件放到以下目录:")
        print(f"  {self.output_dir.absolute()}\n")

        print("步骤5: 运行解压命令:")
        print(f"  python scripts/extract_modelscope.py --input {self.output_dir}/下载的文件名.zip\n")

        # 生成快速下载脚本
        self._create_download_helper(dataset_info)

        print("=" * 60)
        print("\n已生成下载辅助文件:")
        print(f"  {self.output_dir / 'download_links.txt'}")
        print("=" * 60)

        return False  # 返回False，因为需要手动下载

    def _create_download_helper(self, dataset_info):
        """创建下载辅助文件"""
        helper_file = self.output_dir / "download_links.txt"

        with open(helper_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("井盖数据集下载链接\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"数据集名称: {dataset_info['name']}\n")
            f.write(f"数据集ID: {dataset_info['dataset_id']}\n")
            f.write(f"描述: {dataset_info['description']}\n\n")

            f.write("下载链接:\n")
            f.write(f"{dataset_info['url']}\n\n")

            f.write("使用说明:\n")
            f.write("1. 复制上面的链接到浏览器\n")
            f.write("2. 登录ModelScope账号\n")
            f.write("3. 下载数据集压缩包\n")
            f.write("4. 将文件放到此目录\n")
            f.write("5. 运行: python scripts/extract_modelscope.py\n\n")

        print(f"✅ 已创建下载辅助文件: {helper_file}")

    def download_all_datasets(self):
        """下载所有可用数据集"""
        print("开始下载所有井盖数据集...\n")

        results = {}
        for key in DATASETS.keys():
            print(f"\n{'='*60}")
            print(f"正在处理: {key}")
            print(f"{'='*60}")
            results[key] = self.download_modelscope_dataset(key)

        print("\n" + "=" * 60)
        print("下载任务汇总")
        print("=" * 60)
        for key, success in results.items():
            status = "✅ 成功" if success else "⚠️ 需手动"
            print(f"{key}: {status}")


def extract_downloaded_datasets(input_dir="data/raw", output_dir="data/raw"):
    """
    解压已下载的数据集文件

    Args:
        input_dir: 下载文件所在目录
        output_dir: 解压输出目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 查找所有zip文件
    zip_files = list(input_path.glob("*.zip"))
    zip_files.extend(input_path.glob("*.ZIP"))

    if not zip_files:
        print(f"❌ 在 {input_dir} 中未找到zip文件")
        return

    print(f"找到 {len(zip_files)} 个zip文件")

    for zip_file in zip_files:
        print(f"\n处理: {zip_file.name}")

        # 创建解压目录（以zip文件名命名）
        extract_dir = output_path / zip_file.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 解压
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"包含 {len(file_list)} 个文件")

            for file in tqdm(file_list, desc=f"解压 {zip_file.name}"):
                zip_ref.extract(file, extract_dir)

        print(f"✅ 已解压到: {extract_dir}")

        # 检查是否需要进一步解压内部zip
        inner_zips = list(extract_dir.glob("**/*.zip"))
        if inner_zips:
            print(f"发现 {len(inner_zips)} 个内部zip文件，继续解压...")
            for inner_zip in inner_zips:
                inner_extract_dir = inner_zip.parent
                with zipfile.ZipFile(inner_zip, 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        zip_ref.extract(file, inner_extract_dir)
                inner_zip.unlink()  # 删除内部zip
                print(f"  已解压: {inner_zip.name}")

    print("\n✅ 所有文件解压完成")


def create_kaggle_downloader():
    """创建Kaggle数据集下载脚本"""
    script_content = '''#!/usr/bin/env python3
"""
Kaggle数据集下载辅助脚本
需要先配置kaggle API密钥
"""

import os
from pathlib import Path


def setup_kaggle_api():
    """配置Kaggle API"""
    print("Kaggle API配置指南")
    print("=" * 50)
    print("1. 登录Kaggle: https://www.kaggle.com/")
    print("2. 进入账号设置 -> API -> Create New API Token")
    print("3. 下载kaggle.json文件")
    print("4. 将kaggle.json放到 ~/.kaggle/ 目录")
    print("=" * 50)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        print("✅ Kaggle API已配置")
        return True
    else:
        print("❌ 未找到kaggle.json")
        print(f"请将kaggle.json放到: {kaggle_dir}")
        return False


def download_kaggle_dataset(dataset_name, output_dir="data/raw"):
    """
    下载Kaggle数据集

    Args:
        dataset_name: 数据集名称，如 "username/dataset-name"
        output_dir: 输出目录
    """
    try:
        import kaggle
    except ImportError:
        print("正在安装kaggle...")
        os.system("pip install kaggle -q")
        import kaggle

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"正在下载数据集: {dataset_name}")

    kaggle.api.dataset_download_files(
        dataset_name,
        path=str(output_path),
        unzip=True
    )

    print(f"✅ 下载完成: {output_path}")


def search_manhole_datasets():
    """搜索井盖相关数据集"""
    print("\\n搜索Kaggle上的井盖数据集...")
    print("建议搜索关键词:")
    print("  - manhole")
    print("  - sewer")
    print("  - road damage")
    print("  - infrastructure")
    print("\\n访问: https://www.kaggle.com/datasets")


if __name__ == "__main__":
    if setup_kaggle_api():
        # 示例：下载特定数据集
        # download_kaggle_dataset("dataset-name/here", "data/raw")
        search_manhole_datasets()
'''

    script_path = Path(__d/jglw/yolov11-manhole-detection/scripts/download_kaggle.py')
    script_path.write_text(script_content, encoding='utf-8')
    print(f"✅ 已创建Kaggle下载脚本: {script_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='ModelScope数据集下载工具')
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()) + ['all'],
                       default='manhole_basic', help='数据集名称')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='输出目录')
    parser.add_argument('--extract-only', action='store_true',
                       help='仅解压已有文件')

    args = parser.parse_args()

    downloader = ModelScopeDownloader(args.output)

    if args.extract_only:
        # 仅解压模式
        extract_downloaded_datasets(args.output, args.output)
    elif args.dataset == 'all':
        # 下载全部
        downloader.download_all_datasets()
    else:
        # 下载指定数据集
        downloader.download_modelscope_dataset(args.dataset)

    # 创建Kaggle下载脚本
    create_kaggle_downloader()

    print("\n" + "=" * 60)
    print("下一步操作:")
    print("=" * 60)
    print("1. 如需手动下载，请查看 data/raw/download_links.txt")
    print("2. 下载完成后，运行:")
    print("   python scripts/extract_modelscope.py")
    print("3. 运行数据预处理:")
    print("   python scripts/prepare_data.py --raw_dir data/raw/解压后的目录")
    print("=" * 60)


if __name__ == '__main__':
    main()
