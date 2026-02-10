# 井盖检测公开数据集汇总

**更新日期**: 2026-02-08
**状态**: ✅ 已确认多个可用数据集

---

## 🎯 推荐数据集（按质量排序）

### ⭐ 推荐1：ModelScope下水井盖数据集

| 属性 | 值 |
|------|---|
| 数据量 | **10,500张** |
| 格式 | Pascal VOC + YOLO |
| 类别 | 下水井盖 |
| 平台 | [ModelScope](https://www.modelscope.cn/datasets/xisowei666/xyxr_datasets) |
| 优势 | 数据量大，格式标准 |

**下载方式**:
```bash
# 安装ModelScope SDK
pip install modelscope

# Python下载数据集
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('xisowei666/xyxr_datasets', split='train')
```

---

### ⭐ 推荐2：城市街道井盖数据集（腾讯云）

| 属性 | 值 |
|------|---|
| 数据量 | **4,404张** |
| 格式 | Pascal VOC + YOLO |
| 类别 | 5类 (broke等) |
| 总框数 | 5,321个 |
| 来源 | [腾讯云开发者社区](https://cloud.tencent.com/developer/article/2544838) |

**类别说明**: broke(破损)、uncovered(未盖)、missing(丢失)等

---

### ⭐ 推荐3：YOLO井盖缺陷数据集

| 属性 | 值 |
|------|---|
| 数据量 | **2,000张** |
| 格式 | YOLO格式(txt标签) |
| 划分 | train/val/test已划分 |
| 配置 | 附data.yaml |
| 来源 | [CSDN](https://blog.csdn.net/2403_88275621/article/details/155691501) |

**优势**: 开箱即用，支持YOLOv5/v8

---

### 推荐4：井盖隐患数据集（YOLO V8）

| 属性 | 值 |
|------|---|
| 数据量 | **1,288张** |
| 应用 | YOLO V8目标检测 |
| 来源 | [知乎](https://zhuanlan.zhihu.com/p/692013412) |

---

### 推荐5：道路表面缺陷数据集（含井盖）

| 属性 | 值 |
|------|---|
| 数据量 | **6,000张**高分辨率图片 |
| 类别 | 裂缝、井盖、坑洼、修补区域 |
| 格式 | YOLO标准格式 |
| 来源 | [掘金](https://juejin.cn/post/7539858904034787367) |

---

## 🆕 新增数据集 (Roboflow Universe - 2024更新)

### ⭐ 推荐0：Roboflow SideSeeing Manhole (NEW!)

| 属性 | 值 |
|------|---|
| 数据量 | **1,427张** |
| 格式 | YOLO |
| 类别 | **4类**: Broken, Loose, Uncovered, Good |
| 平台 | [Roboflow Universe](https://universe.roboflow.com/sideseeing/manhole-cover-dataset-yolo-62sri) |
| 优势 | 类别清晰，有预训练模型，可直接YOLO训练 |

### 推荐0b：Roboflow Manhole 5K Images (NEW!)

| 属性 | 值 |
|------|---|
| 数据量 | **5,000张** |
| 格式 | YOLO |
| 平台 | [Roboflow Universe](https://universe.roboflow.com/create-dataset-for-yolo/manhole-cover-dataset-5k-images) |
| 优势 | 大规模，适合baseline训练 |

### 推荐0c：Roboflow Road Damage Manhole (NEW!)

| 属性 | 值 |
|------|---|
| 数据量 | **990张** |
| 格式 | YOLO |
| 更新日期 | **2024年8月20日** (最新!) |
| 平台 | [Roboflow Universe](https://universe.roboflow.com/hazels-space/road-damage-manhole-sewers-covers/dataset/8) |
| 优势 | 最新更新，场景多样化 |

**Roboflow下载方法**:
```bash
# 安装roboflow
pip install roboflow

# Python下载示例
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("sideseeing").project("manhole-cover-dataset-yolo-62sri")
dataset = project.version(1).download("yolov8")
```

---

## 📊 数据集对比 (更新)

| 数据集 | 数量 | 格式 | 类别数 | 更新时间 | 推荐度 |
|--------|------|------|--------|----------|--------|
| **Roboflow SideSeeing** | 1,427张 | YOLO | 4 | - | ⭐⭐⭐⭐⭐ |
| **Roboflow 5K Images** | 5,000张 | YOLO | 多类 | - | ⭐⭐⭐⭐⭐ |
| **Roboflow Road Damage** | 990张 | YOLO | 多类 | 2024-08 | ⭐⭐⭐⭐ |
| ModelScope下水井盖 | 10,500张 | VOC+YOLO | 1 | - | ⭐⭐⭐⭐⭐ |
| 城市街道井盖 | 4,404张 | VOC+YOLO | 5 | - | ⭐⭐⭐⭐⭐ |
| Kaggle Manhole | TBD | YOLOv8 | 多类 | - | ⭐⭐⭐ |
| YOLO井盖缺陷 | 2,000张 | YOLO | 多类 | - | ⭐⭐⭐⭐ |
| 井盖隐患 | 1,288张 | YOLO | 多类 | - | ⭐⭐⭐ |

---

## 🔗 快速下载链接

| 平台 | 链接 |
|------|------|
| **ModelScope** | https://www.modelscope.cn/datasets/xisowei666/xyxr_datasets |
| **CSDN数据集1** | https://blog.csdn.net/2401_86822270/article/details/144759708 |
| **CSDN数据集2** | https://blog.csdn.net/2403_88275621/article/details/155691501 |
| **腾讯云** | https://cloud.tencent.com/developer/article/2544838 |
| **知乎** | https://zhuanlan.zhihu.com/p/692013412 |
| **掘金** | https://juejin.cn/post/7539858904034787367 |

---

## 💡 使用建议

### 首选方案：ModelScope数据集
- 数据量最大（10,500张）
- 格式标准（VOC+YOLO）
- 官方平台可靠

### 备选方案：城市街道井盖数据集
- 类别丰富（5类）
- 适合细粒度分类
- 标注质量较好

### 补充方案：多个数据集合并
- 合并多个数据集
- 统一标注格式
- 增加数据多样性

---

## ⚠️ 注意事项

1. **类别映射**: 不同数据集类别定义不同，需要统一映射
2. **标注质量**: 下载后需检查标注质量
3. **版权许可**: 注意数据集的使用许可
4. **格式转换**: 部分数据集可能需要格式转换

---

## 📝 数据集记录模板

下载后填写 `data/dataset_candidates.json`:

```json
{
  "dataset_name": "城市街道井盖数据集",
  "source": "腾讯云开发者社区",
  "download_url": "https://cloud.tencent.com/developer/article/2544838",
  "total_images": 4404,
  "resolution": "1024x1024",
  "format": "VOC + YOLO",
  "has_annotations": true,
  "annotation_format": "txt (YOLO format)",
  "classes": ["intact", "broke", "uncovered", "missing", "other"],
  "license": "开源",
  "download_date": "2026-02-07",
  "notes": "5类井盖状态，标注完整"
}
```

---

**Sources:**
- [如何用yolov8训练使用井盖检测缺陷数据集 - CSDN](https://blog.csdn.net/2401_86822270/article/details/144759708)
- [ModelScope数据集平台](https://www.modelscope.cn/datasets/xisowei666/xyxr_datasets)
- [城市街道井盖破损未盖丢失数据集 - 腾讯云](https://cloud.tencent.com/developer/article/2544838)
- [基于YOLO V8的高精度井盖隐患检测识别系统 - 知乎](https://zhuanlan.zhihu.com/p/692013412)
- [井盖缺陷数据集 - CSDN](https://blog.csdn.net/2403_88275621/article/details/155691501)
- [道路表面缺陷数据集 - 掘金](https://juejin.cn/post/7539858904034787367)
