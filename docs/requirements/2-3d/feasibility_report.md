# 📊 SHARP 2D→3D 空间照片集成可行性报告

> **版本:** 1.0 · 2026-02-18
>
> 本文档评估将 Apple 开源项目 **SHARP**（[apple/ml-sharp](https://github.com/apple/ml-sharp)）集成到 iPhotron 本地相册管理器中的可行性，实现从单张 2D 照片生成 3D 高斯溅射（3D Gaussian Splatting）模型，并在小角度范围内模拟摄像头移动，提供类似 macOS 空间照片的视觉效果。

---

## 目录

1. [项目概述](#1-项目概述)
2. [SHARP 技术分析](#2-sharp-技术分析)
3. [集成可行性评估](#3-集成可行性评估)
4. [Release 打包可行性评估](#4-release-打包可行性评估)
5. [非 NVIDIA 设备兼容性评估](#5-非-nvidia-设备兼容性评估)
6. [许可证兼容性分析](#6-许可证兼容性分析)
7. [风险评估与缓解措施](#7-风险评估与缓解措施)
8. [综合结论与建议](#8-综合结论与建议)

---

## 1. 项目概述

### 1.1 SHARP 简介

SHARP（**Sharp Monocular View Synthesis**）是 Apple 于 2025 年开源的单目视图合成模型，核心能力为：

| 特性 | 说明 |
|------|------|
| **输入** | 单张 2D 照片 |
| **输出** | 3D 高斯溅射（3DGS）场景表示（`.ply` 文件） |
| **推理速度** | 标准 GPU 上不到 1 秒 |
| **渲染** | 3DGS 实时渲染，支持小范围视角变化 |
| **度量空间** | 输出带有绝对尺度，支持度量级摄像头移动 |
| **论文** | [arXiv:2512.10685](https://arxiv.org/abs/2512.10685) |

### 1.2 iPhotron 当前技术栈

| 组件 | 技术 |
|------|------|
| **语言** | Python 3.12+ |
| **GUI** | PySide6 (Qt6) |
| **GPU 渲染** | OpenGL 3.3 (PyOpenGL) |
| **图像处理** | Pillow, OpenCV, NumPy, Numba |
| **架构** | MVVM + DDD 分层架构 |
| **打包方式** | setuptools / pip |

### 1.3 目标效果

实现类似 macOS 空间照片的交互式 3D 视差效果：
- 用户选择一张照片后，可进入"空间照片"模式
- 通过鼠标移动/设备倾斜在小角度范围（约 ±5°）内模拟摄像头平移
- 产生自然的视差和深度感，物体前后景有不同的位移量

---

## 2. SHARP 技术分析

### 2.1 架构概览

```
输入照片 (RGB)
    │
    ▼
┌─────────────────────────┐
│  MonodepthWithEncoding  │  ← 单目深度估计 + 特征编码
│    Adaptor (timm)       │     (基于 timm 预训练视觉模型)
└────────┬────────────────┘
         │  深度图 + 编码特征
         ▼
┌─────────────────────────┐
│     Initializer         │  ← 深度归一化 + 基础高斯参数初始化
└────────┬────────────────┘
         │  基础高斯 + 特征
         ▼
┌─────────────────────────┐
│    Feature Model        │  ← 主网络: 预测高斯参数的增量
│   + Prediction Head     │
└────────┬────────────────┘
         │  增量参数
         ▼
┌─────────────────────────┐
│   GaussianComposer      │  ← 合成最终高斯参数 (位置/颜色/协方差/不透明度)
└────────┬────────────────┘
         │
         ▼
  Gaussians3D (3DGS PLY)
```

### 2.2 核心依赖

| 依赖 | 版本 | 用途 | 与 iPhotron 关系 |
|------|------|------|------------------|
| **PyTorch** | 2.8.0 | 深度学习推理框架 | 🆕 需新增 |
| **torchvision** | 0.23.0 | 图像预处理 | 🆕 需新增 |
| **timm** | 1.0.20 | 预训练视觉模型 (DINOv2 等) | 🆕 需新增 |
| **gsplat** | 1.5.3 | 3DGS 渲染 (仅 CUDA) | 🆕 需新增 (可选) |
| **plyfile** | 1.1.2 | PLY 文件读写 | 🆕 需新增 |
| **scipy** | 1.16.2 | 数学计算 | 🆕 需新增 |
| **Pillow** | 11.x | 图像读取 | ✅ 已有 |
| **pillow-heif** | 1.1.1 | HEIC 支持 | ✅ 已有 |
| **NumPy** | 2.3.x | 数组运算 | ✅ 已有 |

### 2.3 模型推理流程

```python
# 1. 加载模型 (首次自动下载 ~500MB checkpoint)
predictor = create_predictor(PredictorParams())
predictor.load_state_dict(state_dict)
predictor.eval()
predictor.to(device)  # "cuda" / "mps" / "cpu"

# 2. 输入图片预处理
image_pt = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
image_resized = F.interpolate(image_pt[None], size=(1536, 1536))

# 3. 推理 → 3DGS
gaussians_ndc = predictor(image_resized, disparity_factor)
gaussians = unproject_gaussians(gaussians_ndc, ...)

# 4. 保存/渲染
save_ply(gaussians, f_px, (h, w), output_path)
```

### 2.4 设备支持情况

| 操作 | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU |
|------|:---:|:---:|:---:|
| **模型推理** (predict) | ✅ 最快 (<1秒) | ✅ 支持 (~数秒) | ✅ 支持 (较慢) |
| **3DGS 渲染** (render via gsplat) | ✅ 唯一支持 | ❌ 不支持 | ❌ 不支持 |

> **关键发现：** SHARP 的模型推理（从 2D 照片生成 3DGS 参数）支持 CUDA、MPS 和 CPU 三种设备。但其内置的 gsplat 视频渲染器仅支持 CUDA。

---

## 3. 集成可行性评估

### 3.1 技术集成方案

**结论：✅ 可行**

SHARP 可以作为 Python 子模块集成到 iPhotron 中。推荐的集成策略：

#### 方案一：子进程调用方式（推荐初期采用）

```
iPhotron GUI → subprocess.Popen("sharp predict -i photo.jpg -o output/") → 读取 .ply
```

- **优点**：零侵入，SHARP 作为独立 CLI 工具，依赖隔离
- **缺点**：需要用户单独安装 SHARP 环境，跨进程通信开销

#### 方案二：库级别集成（推荐最终方案）

```
iPhotron → import sharp → predictor(image) → Gaussians3D → 自研 OpenGL 渲染器
```

- **优点**：无进程间通信，GPU 内存共享，推理+渲染一体化
- **缺点**：增加 PyTorch 等大型依赖

#### 推荐路径

1. **Phase 1**：以可选依赖（`pip install iPhoto[3d]`）方式引入 SHARP
2. **Phase 2**：在 iPhotron 现有 OpenGL 渲染管线中实现 3DGS 渲染器（替代 gsplat 的 CUDA 限制）
3. **Phase 3**：优化用户交互体验（鼠标/陀螺仪驱动视角）

### 3.2 渲染方案

**核心问题**：gsplat 渲染器仅支持 CUDA，iPhotron 使用 OpenGL 3.3。

**解决方案**：自研 OpenGL 3DGS 渲染器

| 方案 | 可行性 | 说明 |
|------|:---:|------|
| **OpenGL 3.3 Compute Shader** | ❌ | OpenGL 3.3 不支持 Compute Shader (需要 4.3+) |
| **OpenGL 3.3 Fragment Shader** | ✅ | 通过分层排序 + Alpha Blending 实现简化版 3DGS 渲染 |
| **升级到 OpenGL 4.3+** | ✅ | 支持 Compute Shader，可实现高效的 tile-based 3DGS 渲染 |
| **使用 Vulkan** | ⚠️ | 跨平台最优方案，但需重构渲染管线 |
| **预渲染视角序列** | ✅ | 在 CPU/GPU 上预先渲染有限视角帧，运行时插值 |

**推荐渲染方案**：

**Phase 1 — 预渲染 + 纹理切换**（最快实现）
- 在后台用 PyTorch 预渲染多角度视图（如 3×3 = 9 个方向），存为纹理
- 运行时根据鼠标位置，在现有 OpenGL 管线中对纹理进行插值混合
- 无需 CUDA，CPU/MPS 均可用
- 效果有限但足以实现基础的空间照片视差

**Phase 2 — OpenGL Fragment Shader 3DGS 渲染**（高质量方案）
- 实现基于 Fragment Shader 的高斯溅射渲染
- 利用 iPhotron 已有的 `gl_renderer.py`、`gl_shader_manager.py`、`gl_texture_manager.py` 基础设施
- 支持实时交互、任意视角

### 3.3 架构集成点

SHARP 功能在 iPhotron 现有分层架构中的位置：

```
src/iPhoto/
├── ai/                         # AI 子系统（已有人脸/OCR 规划）
│   └── spatial/                # 🆕 空间照片子模块
│       ├── sharp_predictor.py  #     SHARP 推理封装
│       ├── gaussian_renderer.py#     OpenGL 3DGS 渲染器
│       └── spatial_cache.py    #     3DGS 缓存管理
├── domain/
│   └── models/
│       └── spatial_photo.py    # 🆕 空间照片领域模型
├── application/
│   └── use_cases/
│       └── generate_spatial.py # 🆕 生成空间照片用例
├── gui/
│   └── ui/
│       └── widgets/
│           └── gl_spatial_viewer.py  # 🆕 空间照片查看器
```

### 3.4 与现有功能的协同

| 现有功能 | 协同点 |
|----------|--------|
| **OpenGL 渲染管线** | 复用 `gl_renderer.py`、Shader 管理器、纹理管理器 |
| **GPU Pipeline** | 复用 `ShaderPrecompiler`、`FBOPool` |
| **后台任务** | 复用 `QRunnable` 任务框架进行 3DGS 生成 |
| **缓存系统** | 复用 SQLite 缓存架构存储生成的 3DGS |
| **图像加载** | 复用 Pillow/pillow-heif 加载 HEIC/JPG |
| **AI 子系统** | 与人脸/OCR 共享 GPU/CPU 后端检测逻辑 |

---

## 4. Release 打包可行性评估

### 4.1 打包方案

**结论：⚠️ 有条件可行**

| 方面 | 评估 | 说明 |
|------|:---:|------|
| **pip 安装** | ✅ | 可作为可选依赖 `pip install iPhoto[3d]` |
| **PyInstaller/cx_Freeze** | ⚠️ | PyTorch 体积大 (>1GB)，需特殊配置 |
| **应用体积** | ⚠️ | 模型 checkpoint ~500MB + PyTorch ~800MB = 额外 ~1.5GB |
| **首次运行** | ⚠️ | 模型会自动下载缓存，需网络连接 |

### 4.2 推荐打包策略

```
iPhoto-base.whl           # 基础功能 (~50MB)
iPhoto-3d.whl             # 3D 扩展 (仅声明依赖)
├── torch                  # ~800MB (按平台动态选择 CPU/CUDA/MPS)
├── torchvision            # ~50MB
├── timm                   # ~5MB
├── sharp                  # ~2MB (模型代码)
└── model checkpoint       # ~500MB (延迟下载)
```

#### 分层安装方案

```bash
# 基础安装 (不含 3D 功能)
pip install iPhoto

# 安装 3D 功能 (CPU 版本, 最小体积)
pip install iPhoto[3d-cpu]

# 安装 3D 功能 (CUDA 版本, 最佳性能)
pip install iPhoto[3d-cuda]

# 安装 3D 功能 (MPS 版本, macOS Apple Silicon)
pip install iPhoto[3d-mps]
```

### 4.3 Release 体积预估

| 配置 | 预估体积 |
|------|---------|
| iPhoto 基础版 | ~50MB |
| iPhoto + 3D (CPU) | ~1.2GB |
| iPhoto + 3D (CUDA) | ~2.5GB |
| 模型 checkpoint (延迟下载) | ~500MB |

### 4.4 桌面应用打包

| 打包工具 | 可行性 | 说明 |
|----------|:---:|------|
| **PyInstaller** | ⚠️ | 可行但产物体积大 (~3GB+)，需排除不必要的 CUDA 库 |
| **Nuitka** | ⚠️ | 编译后性能更好，但 PyTorch 兼容性需验证 |
| **conda/mamba** | ✅ | 最佳依赖管理方案，可精确控制 PyTorch 版本 |
| **Docker** | ✅ | 适合服务端部署，桌面体验不佳 |

**推荐**：使用 `pip + optional dependencies` 分发，用户按需安装 3D 扩展。

---

## 5. 非 NVIDIA 设备兼容性评估

### 5.1 设备支持矩阵

| 设备类型 | 推理 (predict) | 渲染 (自研 OpenGL) | 渲染 (gsplat) | 综合评估 |
|----------|:---:|:---:|:---:|:---:|
| **NVIDIA GPU (CUDA)** | ✅ 最优 (<1s) | ✅ | ✅ | ✅ 完全支持 |
| **Apple Silicon (MPS)** | ✅ 支持 (~3-5s) | ✅ | ❌ | ✅ 可行 (自研渲染) |
| **AMD GPU (ROCm)** | ⚠️ PyTorch ROCm | ✅ | ❌ | ⚠️ 有限支持 |
| **Intel GPU (XPU)** | ⚠️ PyTorch XPU | ✅ | ❌ | ⚠️ 有限支持 |
| **纯 CPU** | ✅ 支持 (~15-30s) | ✅ | ❌ | ⚠️ 可用但较慢 |

### 5.2 关键结论

1. **模型推理不依赖 NVIDIA**：SHARP 的 `predict` 功能支持 CUDA、MPS 和 CPU。PyTorch 自身的跨平台支持使得推理部分天然兼容多种硬件。

2. **gsplat 渲染器仅限 CUDA**：SHARP 自带的 gsplat 视频渲染功能要求 NVIDIA GPU。但此渲染器仅用于生成演示视频，**不影响核心的 3DGS 生成能力**。

3. **自研 OpenGL 渲染器是关键**：通过在 iPhotron 现有 OpenGL 管线中实现 3DGS 渲染，可以完全绕过 gsplat 的 CUDA 限制，在所有支持 OpenGL 3.3+ 的设备上实现实时交互。

4. **CPU 模式可用**：即使没有任何 GPU，用户仍可使用 CPU 进行推理（较慢但可用），配合预渲染方案实现基础的空间照片效果。

### 5.3 各平台性能预估

| 硬件 | 推理时间 | 渲染 FPS (OpenGL) | 用户体验 |
|------|---------|------------------|----------|
| RTX 3060 (CUDA) | <1s | 60+ fps | 🟢 流畅 |
| M1/M2 Mac (MPS) | 3-5s | 60+ fps | 🟢 良好 |
| AMD RX 6600 (ROCm) | 2-5s | 60+ fps | 🟡 可用 |
| Intel Core i7 (CPU) | 15-30s | 30+ fps | 🟡 可用（推理等待较长） |
| Intel Core i5 (CPU) | 30-60s | 20+ fps | 🔴 勉强可用 |

---

## 6. 许可证兼容性分析

### 6.1 SHARP 许可证

SHARP 项目包含两份许可证：

| 许可证 | 覆盖范围 | 类型 |
|--------|---------|------|
| **LICENSE** (Apple Software License) | 源代码 | 类 BSD，允许修改和再分发 |
| **LICENSE_MODEL** (Research Model License) | 模型权重 | **仅限研究用途**，禁止商业使用 |

### 6.2 关键条款分析

#### 代码许可证 (LICENSE)

> "Apple grants you a personal, non-exclusive license... to use, reproduce, modify and redistribute the Apple Software, with or without modifications, in source and/or binary forms"

- ✅ 可以修改和再分发源代码
- ⚠️ 如果完整再分发需保留版权声明
- ❌ 不得使用 Apple 名称/商标进行推广

#### 模型许可证 (LICENSE_MODEL)

> "License Scope: ... exclusively for Research Purposes."
>
> "Research Purposes" means non-commercial scientific research and academic development activities... **does not include any commercial exploitation, product development or use in any commercial product or service.**

- ❌ **模型权重仅限研究用途**
- ❌ **不可用于商业产品**
- ⚠️ 如果 iPhotron 是开源非商业项目，则可用

### 6.3 许可证兼容性结论

| 场景 | 是否合规 | 说明 |
|------|:---:|------|
| iPhotron 作为开源研究项目分发 | ✅ | 符合"研究用途"定义 |
| iPhotron 作为免费开源软件分发（非商业） | ⚠️ | 需确认是否满足"研究用途"的严格定义 |
| iPhotron 作为商业产品销售 | ❌ | 违反模型许可证 |
| 仅集成 SHARP 代码，使用自训练模型 | ✅ | 代码许可证允许修改和再分发 |

### 6.4 建议

1. 如果 iPhotron 为**非商业开源项目**：可以直接集成 SHARP 代码和模型
2. 如果 iPhotron 有**任何商业化计划**：
   - 仅使用 SHARP 的代码架构（代码许可证允许）
   - 自行训练或使用兼容许可证的替代模型
   - 或联系 Apple 获取商业授权

---

## 7. 风险评估与缓解措施

### 7.1 技术风险

| 风险 | 等级 | 缓解措施 |
|------|:---:|---------|
| PyTorch 引入导致应用体积膨胀 | 🟡 中 | 使用 optional dependencies，延迟下载模型 |
| OpenGL 3DGS 渲染器开发难度高 | 🟡 中 | Phase 1 先用预渲染方案，逐步实现实时渲染 |
| CPU 推理速度慢影响用户体验 | 🟡 中 | 异步后台推理 + 进度指示 + 缓存已生成的 3DGS |
| 不同平台 PyTorch 兼容性 | 🟢 低 | PyTorch 官方支持主流平台 |
| 模型质量在某些场景下效果不佳 | 🟡 中 | 仅在效果可接受时显示 3D 功能按钮 |

### 7.2 非技术风险

| 风险 | 等级 | 缓解措施 |
|------|:---:|---------|
| 模型许可证限制商业化 | 🔴 高 | 明确项目定位，必要时自训练模型 |
| SHARP 项目不再维护 | 🟡 中 | Fork 代码，仅依赖推理部分 |
| 用户硬件不满足要求 | 🟡 中 | 优雅降级：无 GPU → CPU + 预渲染 |

---

## 8. 综合结论与建议

### 8.1 总体评估

| 评估维度 | 结论 | 信心度 |
|----------|------|:---:|
| **能否实现集成** | ✅ 可行 | 🟢 高 |
| **能否作为整体 Release** | ⚠️ 有条件可行（体积大，建议可选安装） | 🟡 中 |
| **能否运行在非 NVIDIA 设备** | ✅ 可行（推理支持 MPS/CPU，渲染需自研） | 🟢 高 |
| **许可证是否兼容** | ⚠️ 非商业用途可行，商业用途需替代方案 | 🟡 中 |

### 8.2 推荐实施路径

```
Phase 1（MVP，2-3 周）
├── 以 optional dependency 引入 SHARP
├── 实现后台 3DGS 生成任务
├── 预渲染多角度视图 + 纹理插值
└── 基础的空间照片查看器 Widget

Phase 2（增强，3-4 周）
├── OpenGL Fragment Shader 3DGS 渲染器
├── 鼠标交互驱动实时视角变化
├── 3DGS 缓存与管理
└── 设置面板（推理设备选择、质量选项）

Phase 3（优化，2-3 周）
├── 推理性能优化（模型量化、ONNX Runtime）
├── 渲染性能优化（LOD、视锥裁剪）
├── 多平台测试与打包
└── 用户文档
```

### 8.3 最终建议

**建议推进实施**。SHARP 与 iPhotron 的技术栈高度兼容（同为 Python 生态），现有的 OpenGL 渲染基础设施可以复用，且通过自研渲染器可以解决 gsplat 的 CUDA 限制。建议以可选插件的形式集成，避免对现有轻量级安装的影响。

关键成功因素：
1. 自研 OpenGL 3DGS 渲染器的实现质量
2. 合理的缓存策略避免重复推理
3. 优雅的降级方案确保非 GPU 用户也能使用
4. 明确的许可证合规策略
