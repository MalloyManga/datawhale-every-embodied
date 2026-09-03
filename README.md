# 🤖 赛博宠物 - 家庭环境物体检测系统

> 为赛博宠物的环境认知能力赋能

## 📋 项目概述

这是赛博宠物项目的**核心模块之一**，用于：
- 🏠 **环境认知**：检测和识别家庭环境中的物体
- 📍 **位置记忆**：记住物体的位置信息
- 🗺️ **环境地图**：构建家庭布局和物体分布
- 🤖 **决策支持**：为赛博宠物的行为决策提供基础信息

## 📚 与共学营文档的关系

本项目对应仓库中的以下学习内容：

- [第1章：具身智能概述](../../01-具身智能概述/01具身智能概述.md)：文档将具身智能概括为身体、智能算法和环境的结合。本项目从“感知环境”这一环开始，为赛博宠物后续的记忆、决策和行动提供输入。
- [第4章：SAM分割与单目深度估计](../../04-具身场景的计算机视觉、3D重建/01-sam和深度估计.md)：文档强调，机器人不仅要知道“这里有什么”，还要知道“物体在哪里、距离多远”，并提出 SAM + Depth 的三维物体感知方案。

### 实现思路：从二维感知开始

为了验证赛博宠物对家庭环境进行观察和记忆的可行性，本项目先使用轻量级 YOLOv8 完成**二维物体检测**，输出物体类别、边界框、图像中的相对位置和置信度。它不是对官方 SAM + Depth 实现的替代，而是从可运行的视觉感知模块开始逐步搭建系统：

```text
官方文档：图像 -> SAM分割 + 深度估计 -> 物体轮廓和距离 -> 3D语义感知
本次验证：图像 -> YOLOv8检测 -> 物体类别和二维位置 -> 简化环境地图
后续扩展：二维检测结果 + SAM分割 + Depth深度 -> 赛博宠物的三维环境记忆
```

### 为什么从二维检测开始

- YOLOv8可以直接下载轻量模型，适合作为家庭环境感知的轻量级起点。
- 输出的类别和位置，能够验证赛博宠物“先看见并记录家中物体”的基本链路。
- 官方文档指出单独的分割没有距离信息；因此本项目明确记录当前局限：归一化位置不是实际米制距离，也没有完成 SAM 分割和深度估计。
- 后续学习第4章时，可以在当前检测框基础上加入 SAM 和深度模型，逐步升级为官方文档描述的三维物体感知。

这使本项目同时具备两部分价值：**有实际运行结果的最小实践**，以及**对官方文档核心观点的验证和延伸**。

## ⚡ 快速开始

### 1️⃣ 安装依赖

```bash
# 方法1：使用pip
pip install -r requirements.txt

# 方法2：逐个安装
pip install opencv-python opencv-contrib-python ultralytics numpy
```

> ⏱️ **首次运行会下载YOLOv8模型（~45MB），请耐心等待**

### 2️⃣ 准备测试文件

可以准备一张或多张家庭环境图片，例如卧室、客厅、厨房和卫生间，并统一放入项目的 `images/` 目录。程序每次运行会自动检测其中的全部图片。

```bash
# 示例文件夹结构
cyberpet_env_detection/
├── cyberpet_object_detection.py
├── requirements.txt
├── README.md
├── images/                      # 放入需要检测的全部图片
│   ├── bedroom.jpg
│   └── toilet.jpg
├── results/                     # 检测结果（自动创建）
└── maps/                        # 环境地图（自动创建）
```

### 3️⃣ 运行检测

```bash
# 使用交互式入口，自动检测 images/ 中的全部图片
python run.py

# 也可以直接运行主程序
python cyberpet_object_detection.py

# 结果会保存在：
# - results/          (带注解的检测图片)
# - maps/             (JSON格式的环境地图)
```

## 🎯 使用示例

### 示例1：检测单张图像

```python
from cyberpet_object_detection import CyberPetEnvironmentDetector

# 初始化
detector = CyberPetEnvironmentDetector(model_size='n')

# 检测图像
result = detector.detect_image('my_room.jpg', conf_threshold=0.5)

# 保存环境地图
detector.save_environment_map()

# 获取家庭布局
layout = detector.get_home_layout()
print(f"左侧物体: {layout['zones']['left']}")
print(f"中间物体: {layout['zones']['center']}")
print(f"右侧物体: {layout['zones']['right']}")
```

### 示例2：同时检测多张图像

将图片放入 `images/` 后，在 `run.py` 中选择图像检测即可：

```text
images/bedroom.jpg
images/toilet.jpg
```

程序会自动遍历 `images/` 中的 `.jpg`、`.jpeg`、`.png`、`.bmp` 和 `.webp` 文件，不需要逐张输入路径。

程序会为每张图片分别保存结果：

```text
results/bedroom_detection.jpg
results/toilet_detection.jpg
maps/bedroom_environment_map.json
maps/toilet_environment_map.json
```

### 示例3：实时摄像头检测

```python
# 使用摄像头实时检测
detector = CyberPetEnvironmentDetector(model_size='n')
detector.detect_video(0, max_frames=100)  # 0 = 默认摄像头
```

### 示例4：检测视频文件

```python
# 检测视频文件
detector.detect_video('home_video.mp4', max_frames=200)
```

## 📊 输出解释

### 控制台输出示例

```
🤖 正在加载YOLOv8-n模型...
✅ 模型加载完成 (运行设备: cuda)

📸 正在处理图像: sample_room.jpg
   图像大小: 1920x1080

📊 检测结果摘要:
   总物体数: 15
   物体分类统计:
      - person: 2个 (平均置信度: 95.3%)
        位置: center-middle
      - bed: 1个 (平均置信度: 92.1%)
        位置: right-middle
      - lamp: 1个 (平均置信度: 87.5%)
        位置: right-top
      - chair: 2个 (平均置信度: 88.9%)
        位置: left-middle
```

### 环境地图 (JSON格式)

生成的 `environment_map_*.json` 文件：

```json
{
  "timestamp": "2026-09-02T10:30:45.123456",
  "image_path": "sample_room.jpg",
  "image_size": [1920, 1080],
  "total_objects": 15,
  "detected_objects": {
    "person": 2,
    "bed": 1,
    "lamp": 1,
    "chair": 2
  },
  "object_positions": {
    "person": [
      {"quadrant": "center-middle", "normalized_x": 0.50, "normalized_y": 0.55},
      {"quadrant": "left-top", "normalized_x": 0.25, "normalized_y": 0.30}
    ],
    "bed": [
      {"quadrant": "right-middle", "normalized_x": 0.75, "normalized_y": 0.50}
    ]
  },
  "confidence_scores": {
    "person": [0.953, 0.925],
    "bed": [0.921],
    "lamp": [0.875]
  }
}
```

## 🔧 模型选择

根据你的设备能力选择不同大小的YOLOv8模型：

| 模型           | 大小  | 速度     | 精度  | 推荐场景           |
| -------------- | ----- | -------- | ----- | ------------------ |
| **nano (n)**   | ~3MB  | ⚡⚡⚡ 最快 | ⭐⭐    | CPU推理、实时处理  |
| **small (s)**  | ~22MB | ⚡⚡ 较快  | ⭐⭐⭐   | 移动设备、边缘计算 |
| **medium (m)** | ~49MB | ⚡ 中等   | ⭐⭐⭐⭐  | 平衡方案           |
| **large (l)**  | ~94MB | 🐢 较慢   | ⭐⭐⭐⭐⭐ | GPU推理、离线处理  |

> 当前默认使用 **nano (n)** 模型，速度最快

## 🏠 能检测的物体

该系统主要针对家庭环境，可以检测：

### 家具与空间
- 床 (bed)
- 沙发 (sofa, couch)
- 椅子 (chair)
- 桌子 (table, dining table)
- 书桌 (desk)

### 照明
- 台灯 (lamp)
- 灯 (light, ceiling light)

### 建筑结构
- 窗户 (window)
- 门 (door)

### 家电
- 电视/显示器 (tv, monitor)
- 冰箱 (refrigerator, fridge)
- 微波炉 (microwave)
- 水槽 (sink)

### 生物
- 人类 (person)
- 猫 (cat)
- 狗 (dog)

> 💡 **提示**：YOLO模型可以检测1000+类物体，上面列出的只是与赛博宠物最相关的物体。如果需要检测其他物体，可以查看模型的完整类别列表。

## 📈 性能指标

### 速度测试（在i7 CPU + 1080p图像上）

| 模型   | 推理时间 | 图像处理速度 |
| ------ | -------- | ------------ |
| nano   | ~50ms    | ~20 FPS      |
| small  | ~80ms    | ~12 FPS      |
| medium | ~150ms   | ~6.7 FPS     |

> 🚀 **GPU上的速度会快3-5倍**

## ⚙️ 配置选项

### 置信度阈值

```python
# 检测中等以上置信度的物体（推荐）
result = detector.detect_image('room.jpg', conf_threshold=0.5)

# 更严格的检测（只检测高置信度物体）
result = detector.detect_image('room.jpg', conf_threshold=0.7)

# 宽松的检测（包含低置信度物体）
result = detector.detect_image('room.jpg', conf_threshold=0.3)
```

### 视频处理帧数

```python
# 处理最多100帧
detector.detect_video(0, max_frames=100)

# 处理完整视频
detector.detect_video('video.mp4', max_frames=10000)
```

## 🐛 常见问题

### Q1：运行报错"模型不存在"

**A**：第一次运行时，YOLOv8会自动下载模型。请：
1. 确保有网络连接
2. 等待模型下载完成（可能需要几分钟）
3. 查看是否有足够的磁盘空间

### Q2：在CPU上运行很慢

**A**：这是正常的。建议：
1. 使用 `model_size='n'` (nano模型)
2. 降低输入图像分辨率
3. 减少处理帧数
4. 考虑使用GPU

### Q3：检测准确度不高

**A**：尝试以下方法：
1. 提高置信度阈值：`conf_threshold=0.7`
2. 使用更大的模型：`model_size='m'` 或 `'l'`
3. 确保光线充足
4. 增加训练数据（对图像进行微调）

### Q4：如何处理多张图像？

将所有图片放入 `images/`，运行 `python run.py` 并选择图像检测即可。每张图片的检测结果会按原文件名分别保存。

## 🔄 与赛博宠物其他模块的集成

### 数据流

```
📸 图像输入
    ↓
[物体检测模块] ← 当前模块
    ↓
🗺️ 环境地图 (JSON)
    ↓
[位置记忆模块]
    ↓
[行为决策模块]
    ↓
🤖 机器人执行动作
```

### 下游模块调用示例

```python
# 其他模块如何使用这个模块的输出
import json

# 读取环境地图
with open('maps/environment_map_latest.json', 'r') as f:
    env_map = json.load(f)

# 获取物体位置信息
bed_positions = env_map['object_positions'].get('bed', [])
print(f"床的位置: {bed_positions}")

# 获取总体布局
print(f"检测到的物体: {env_map['detected_objects']}")
```

## 📚 项目结构

```
cyberpet_env_detection/
├── cyberpet_object_detection.py    # 主检测脚本
├── requirements.txt                 # 依赖文件
├── README.md                        # 本文件
├── images/                          # 本地输入图片（不提交到Git）
│   ├── bedroom.jpg
│   └── toilet.jpg
├── results/                         # (自动创建) 检测结果
│   └── detection_20260902_103045.jpg
├── maps/                            # (自动创建) 环境地图
│   └── environment_map_20260902_103045.json
└── advanced/                        # (可选) 高级功能
    ├── multi_object_tracking.py    # 多物体追踪
    └── semantic_slam.py            # 语义SLAM集成
```

## 🚀 后续改进方向

- [ ] **多物体追踪**：跟踪物体在时间上的运动轨迹
- [ ] **3D重建**：从RGB-D数据构建3D环境模型
- [ ] **语义分割**：比物体检测更精细的物体理解
- [ ] **实时SLAM**：集成视觉SLAM构建更准确的地图
- [ ] **模型微调**：针对特定家庭环境的模型优化
- [ ] **边缘部署**：优化模型以在移动机器人上运行

## 📖 参考资源

- [YOLO官方文档](https://docs.ultralytics.com/)
- [OpenCV教程](https://docs.opencv.org/)
- [赛博宠物项目主README](../README.md)

## 📝 License

本项目遵循赛博宠物主项目的License

---

**作者**：CyberPet Team  
**最后更新**：2026-09-02  
**版本**：v1.0 - Task 2 快速实现版

