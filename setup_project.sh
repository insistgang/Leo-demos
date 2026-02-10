#!/bin/bash
# YOLOv11井盖检测项目 - 快速启动脚本
# 日期: 2026-02-08

set -e  # 遇到错误立即退出

echo "========================================="
echo "YOLOv11井盖检测项目 - 快速启动"
echo "========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目路径
PROJECT_DIR="D:\jglw\yolov11-manhole-detection"
cd "$PROJECT_DIR" || exit 1

# 步骤1: 检查Python环境
echo -e "\n${YELLOW}[步骤1/5] 检查Python环境${NC}"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version | awk '{print $2}')
    echo -e "${GREEN}✓ Python版本: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ 未找到Python，请先安装Python 3.10+${NC}"
    exit 1
fi

# 步骤2: 检查conda环境
echo -e "\n${YELLOW}[步骤2/5] 检查conda环境${NC}"
if conda env list | grep -q "^yolov11 "; then
    echo -e "${GREEN}✓ yolov11环境已存在${NC}"
else
    echo "创建yolov11环境..."
    conda env create -f environment.yml
    echo -e "${GREEN}✓ 环境创建完成${NC}"
fi

# 激活环境
echo "激活yolov11环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate yolov11

# 步骤3: 验证PyTorch和CUDA
echo -e "\n${YELLOW}[步骤3/5] 验证PyTorch和CUDA${NC}"
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
"

# 步骤4: 检查数据集
echo -e "\n${YELLOW}[步骤4/5] 检查数据集状态${NC}"
if [ -d "data/processed/images/train" ]; then
    TRAIN_COUNT=$(ls -1 data/processed/images/train/*.jpg 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ 训练集图像: $TRAIN_COUNT 张${NC}"
else
    echo -e "${YELLOW}⚠ 数据集未找到，请运行: python scripts/download_modelscope.py${NC}"
fi

# 步骤5: 显示下一步操作
echo -e "\n${YELLOW}[步骤5/5] 下一步操作${NC}"
echo ""
echo "环境配置完成！请选择下一步："
echo ""
echo "1. 下载数据集:"
echo "   python scripts/download_modelscope.py"
echo ""
echo "2. 检查数据质量:"
echo "   python scripts/check_dataset.py --check_all --base_dir data/processed"
echo ""
echo "3. 启动baseline训练:"
echo "   python scripts/train_baseline.py --config configs/baseline.yaml --mode train"
echo ""
echo "4. 测试YOLOv11推理:"
echo "   yolo detect predict model=yolo11n.pt source=0"
echo ""

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}快速启动脚本执行完成！${NC}"
echo -e "${GREEN}=========================================${NC}"
