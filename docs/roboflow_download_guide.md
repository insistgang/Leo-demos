# Roboflow数据集下载完整指南

> **目标**: 使用API下载SideSeeing Manhole数据集

---

## 📋 步骤1: 获取Roboflow API Key

### 方法1: 从Roboflow网站获取 (推荐)

1. **访问Roboflow**: https://app.roboflow.com/

2. **注册/登录账号**:
   - 点击 "Sign Up" 注册 (免费)
   - 或使用Google/GitHub账号登录

3. **获取API Key**:
   - 登录后访问: https://app.roboflow.com/settings/api
   - 复制你的 "Private API Key"
   - 格式类似: `rf_xxxxxxxxxxxxxxxxxxxxxxx`

### 方法2: 创建环境变量

```bash
# Windows (CMD)
set ROBOFLOW_API_KEY=rf_your_key_here

# Windows (PowerShell)
$env:ROBOFLOW_API_KEY="rf_your_key_here"

# Linux/Mac
export ROBOFLOW_API_KEY=rf_your_key_here
```

---

## 🚀 步骤2: 安装依赖并下载数据集

### 完整执行命令

```bash
# 1. 进入项目目录
cd D:\jglw\yolov11-manhole-detection

# 2. 激活环境 (如果已创建)
conda activate yolov11

# 3. 安装roboflow包
pip install roboflow

# 4. 设置API Key (选择一种方式)
# 方式A: 命令行参数
python scripts/download_roboflow.py --dataset sideseeing --api-key rf_your_key_here

# 方式B: 环境变量
set ROBOFLOW_API_KEY=rf_your_key_here
python scripts/download_roboflow.py --dataset sideseeing
```

---

## 📊 步骤3: 验证下载

```bash
# 检查下载的数据集
dir data\raw\sideseeing

# 应该看到:
# train/
# val/
# test/
# data.yaml
```

---

## 🔧 备用方案: 手动下载

如果API下载失败，可以使用手动下载:

### 手动下载步骤

1. **访问数据集页面**:
   https://universe.roboflow.com/sideseeing/manhole-cover-dataset-yolo-62sri

2. **下载数据集**:
   - 点击页面上的 "Download" 按钮
   - 选择 "YOLOv8" 格式
   - 选择 "Download Dataset to Computer"

3. **解压到项目目录**:
   ```bash
   # 解压下载的zip文件
   # 将内容移动到 data/raw/sideseeing/
   ```

---

## 📁 预期目录结构

下载完成后，目录结构应该是:
```
data/raw/sideseeing/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## ⚠️ 常见问题

### 问题1: API Key无效
```
解决方案:
1. 确认API Key格式为 "rf_" 开头
2. 访问 https://app.roboflow.com/settings/api 重新生成
```

### 问题2: 数据集名称错误
```
解决方案:
1. 确认使用正确的数据集名称: sideseeing
2. 访问数据集页面确认workspace和project名称
```

### 问题3: 网络连接问题
```
解决方案:
1. 检查网络连接
2. 使用手动下载方式
3. 或尝试使用VPN
```

---

## 🎯 快速参考

| 数据集ID | 名称 | 图像数 | 类别 |
|----------|------|--------|------|
| `sideseeing` | SideSeeing Manhole | 1,427 | 4类 |
| `manhole-5k` | Manhole 5K | 5,000 | 多类 |
| `road-damage` | Road Damage | 990 | 多类 |

**推荐**: `sideseeing` - 类别清晰，有预训练模型

---

**准备好下载了吗？运行以下命令开始:**

```bash
cd D:\jglw\yolov11-manhole-detection
pip install roboflow
python scripts/download_roboflow.py --dataset sideseeing --api-key YOUR_API_KEY
```
