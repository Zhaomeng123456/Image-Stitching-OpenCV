# Image-Stitching-OpenCV

基于 SIFT + RANSAC 的双路图像拼接工具，支持本地文件拼接和双 Basler 相机实时拼接。

## 功能

- **本地文件模式**：浏览加载两张图片 → 标定（SIFT 特征匹配 + RANSAC 单应矩阵） → 拼接，支持调参
- **相机采集模式**：连接两台 Basler USB 相机，实时预览 → 抓帧标定 → 实时拼接，FPS 显示
- **标定复用**：标定结果（单应矩阵 H）可保存为 `.npz` 文件，后续拼接跳过 SIFT，大幅提速
- **曝光补偿**：重叠区按通道增益/偏置补偿，消除左右亮度差异
- **自适应混合**：线性渐变权重，平滑过渡
- **可缩放/拖拽图像查看器**：鼠标滚轮缩放、左键拖拽平移，所有图像显示区域均支持

## 技术栈

| 模块 | 依赖 |
|------|------|
| 核心拼接（SIFT / KNN / RANSAC / 混合） | `opencv-python` + `opencv-contrib-python` |
| GUI（tkinter） | Python 内置 |
| Basler 相机 | `pypylon` |
| 图像缩放显示 | `Pillow` |

## 安装

```bash
pip install opencv-python opencv-contrib-python numpy Pillow
# 如果不需要 Basler 相机，可跳过 pypylon
pip install pypylon
```

> 注意：SIFT 在 `opencv-contrib-python` 中。请务必同时安装 `opencv-python` 和 `opencv-contrib-python`。

## 文件结构

```
├── Image_Stitching.py      # 核心拼接类 (Image_Stitching + StitchingRuntime)
├── Image_Stitching_GUI.py  # tkinter GUI 主程序
├── camera_utils.py         # Basler 双相机封装 (BaslerDualCamera)
└── images/                 # 示例图片
```

## 使用

### GUI（推荐）

```bash
python Image_Stitching_GUI.py
```

启动后界面包含两种模式：

1. **本地文件** — 点击左右图按钮选择图片 → 调整参数 → 点击"执行拼接"
2. **相机采集** — 点击切换 → 点击 [重连] 检测相机 → [抓取并标定] 获取 H → [开始实时拼接]

拼接结果、特征匹配图均可在对应标签页中**滚轮缩放 + 拖拽查看**。

### 命令行

```bash
python Image_Stitching.py [左图路径] [右图路径]
```
