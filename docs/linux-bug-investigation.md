# 🐛 Linux 平台行为差异 Bug 根因调查 / Linux Platform Bug Investigation

> 本文档记录 iPhoton 在 Linux 平台发现的 4 个行为不一致 Bug 的详细根因分析，
> 包括具体代码位置、触发条件、平台差异原因以及建议的修复方向。

---

## 目录 / Table of Contents

1. [Bug #1: Gallery View 首张缩略图空白](#bug-1-gallery-view-首张缩略图空白)
2. [Bug #2: Live Photo 配对不正确](#bug-2-live-photo-配对不正确)
3. [Bug #3: 地图聚类返回键显示为黑色实心方块](#bug-3-地图聚类返回键显示为黑色实心方块)
4. [Bug #4: 返回键切换界面后不能正确消失](#bug-4-返回键切换界面后不能正确消失)
5. [总结与优先级 / Summary & Priority](#总结与优先级--summary--priority)

---

## Bug #1: Gallery View 首张缩略图空白

### 现象 / Symptoms

- 在 Linux 下打开 Gallery View 时，**第一张缩略图**有很大概率显示为空白（暗色背景块）
- 点击该空白项可以正确进入 Detail Page，说明数据本身没有问题
- 滚动后再滚回第一项时，缩略图可以正常显示
- Windows 下同样的操作流程不出现此问题

### 涉及文件 / Affected Files

| 文件 | 行号 | 职责 |
|---|---|---|
| `src/iPhoto/gui/ui/widgets/gallery_grid_view.py` | 16-50 | `GalleryViewport` (QOpenGLWidget) 视口与 OpenGL 上下文 |
| `src/iPhoto/gui/ui/widgets/asset_delegate.py` | 76-116 | 缩略图绘制逻辑（`paint()` 方法） |

### 根因分析 / Root Cause Analysis

这是一个**竞态条件**问题，涉及三个因素的交互：

#### 因素 1: OpenGL 上下文初始化时序

`GalleryGridView` 使用 `GalleryViewport`（继承自 `QOpenGLWidget`）作为视口：

```python
# gallery_grid_view.py:16-49
class GalleryViewport(QOpenGLWidget):
    def paintGL(self) -> None:
        self.clear_background()

    def clear_background(self) -> None:
        self.makeCurrent()                # 绑定 OpenGL 上下文
        gl.glClearColor(...)              # 清除背景色
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
```

**平台差异:**
- **Windows:** DWM（Desktop Window Manager）在创建窗口时就准备好了 OpenGL 上下文，
  `makeCurrent()` 在首次 `paintGL()` 调用时总是成功的
- **Linux:** OpenGL 上下文的创建依赖 GLX/EGL，`makeCurrent()` 在首次调用时
  可能尚未完成上下文绑定（尤其在使用 Mesa 软件渲染或特定驱动时）

#### 因素 2: 异步缩略图尚未就绪

缩略图的加载是异步的。当 Grid 首次渲染时，第一个可见项的缩略图可能还在生成中：

```python
# asset_delegate.py:76-80
pixmap = index.data(Qt.DecorationRole)        # 可能返回 None（未就绪）
micro_thumb = None
if not (isinstance(pixmap, QPixmap) and not pixmap.isNull()):
    micro_thumb = index.data(Roles.MICRO_THUMBNAIL)  # 也可能返回 None
```

#### 因素 3: 无图时的回退渲染

当主缩略图和微缩略图均不可用时，delegate 绘制一个纯色矩形作为占位：

```python
# asset_delegate.py:115-116
else:
    painter.fillRect(thumb_rect, QColor("#1b1b1b"))  # 黑色占位块
```

#### 三因素叠加导致 Bug

1. 首次 `paintGL()` 时 OpenGL 上下文在 Linux 上可能未完全就绪
2. 第一个格子项在初始布局时就被绘制（在缩略图异步加载完成之前）
3. 后续项通常在缩略图加载完成后才进入视口，所以不受影响
4. 当缩略图最终加载完成后，会触发 `dataChanged` 信号，
   但在 Linux 上此时 OpenGL 视口可能不会正确触发重绘

### 为何仅 Linux 受影响 / Why Linux-Specific

- Linux 的 OpenGL 上下文初始化（GLX/EGL）比 Windows（WGL/DWM）的异步性更强
- Mesa 驱动的 `makeCurrent()` 在上下文创建完成前的首次调用中可能是无操作
- Qt 的 `QOpenGLWidget` 在 Linux 上的首帧渲染时序与 Windows 不同
- Windows 的 DWM 兼容层使得 OpenGL 上下文更早可用

### 建议修复方向 / Suggested Fix

1. **在 `GalleryViewport.initializeGL()` 中安排延迟更新：**
   在 OpenGL 上下文确认可用后，使用 `QTimer.singleShot(0, self.update)` 强制触发重绘

2. **为 delegate 添加占位图（placeholder pixmap）：**
   在缩略图和微缩略图均不可用时，显示一个加载中动画或品牌占位图，
   而非纯色块。并在数据可用后确保触发视口重绘

3. **在 `GalleryViewport` 中添加 `initializeGL` 回调：**
   ```python
   def initializeGL(self) -> None:
       """Ensure a repaint after the GL context is fully ready."""
       super().initializeGL()
       QTimer.singleShot(0, self.update)
   ```

---

## Bug #2: Live Photo 配对不正确

### 现象 / Symptoms

- Live Photo 的静态图片与动态视频未能正确配对
- 某些 Live Photo 在界面上未显示 Live 标记
- 某些配对关系错误（错误的视频关联到了错误的图片）

### 涉及文件 / Affected Files

| 文件 | 行号 | 职责 |
|---|---|---|
| `src/iPhoto/core/pairing.py` | 70-129 | 三阶段 Live Photo 配对算法 |
| `src/iPhoto/core/pairing.py` | 107-116 | 第二阶段：文件名 stem 匹配 |
| `src/iPhoto/gui/viewmodels/asset_dto_converter.py` | 146-154 | Live Photo 检测（DTO 转换） |

### 根因分析 / Root Cause Analysis

#### 问题 A: 文件名 stem 匹配的大小写敏感性

配对算法的第二阶段使用文件名 stem（无扩展名的文件名）进行匹配：

```python
# pairing.py:111-112
stem = Path(photo["rel"]).stem           # 例如 "IMG_1234"
candidates = [v for v in videos.values()
              if Path(v["rel"]).stem == stem]  # 字符串直接比较
```

**问题:**
- `Path.stem` 的比较是**大小写敏感**的
- iOS 导出的文件可能有混合大小写：`IMG_1234.HEIC` + `img_1234.MOV`
- 在 Linux 的区分大小写文件系统上，`"IMG_1234" != "img_1234"` → 匹配失败
- 在 Windows/macOS 的不区分大小写文件系统上，文件系统会自动规范化，
  但 Python 的 `Path.stem` 仍然是大小写敏感的

**影响范围:** 此问题实际上**在所有平台都存在**，只是在 Linux 上更容易暴露，
因为 Linux 文件系统存储文件名时保留原始大小写且路径比较区分大小写。

#### 问题 B: 目录边界限制

第三阶段的目录临近匹配过于严格：

```python
# pairing.py:122-123
folder = str(Path(photo["rel"]).parent)
candidates = [v for v in videos.values()
              if str(Path(v["rel"]).parent) == folder]
```

当图片和视频分布在不同子目录时（例如用户手动整理过的库），
此匹配策略会失败。

#### 问题 C: DTO 转换中的 Live Photo 检测逻辑

`asset_dto_converter.py` 中有一套独立的 Live Photo 检测逻辑：

```python
# asset_dto_converter.py:146-154
is_live = (mt == "live") or (asset.live_photo_group_id is not None)
if is_video and asset.live_photo_group_id is not None:
    is_live = False
if not is_live and asset.metadata:
    live_partner = asset.metadata.get("live_partner_rel")
    live_role = asset.metadata.get("live_role")
    if live_partner and live_role != 1 and not is_video:
        is_live = True
```

这里 `live_role != 1` 的条件排除了 `live_role == 1` 的资产。
当配对算法因为 stem 不匹配而未能正确设置 `live_role` 时，
此检测也会产生错误结果。配对失败导致下游检测失败，形成连锁效应。

### 为何 Linux 更易暴露 / Why More Visible on Linux

- Linux 文件系统（ext4 等）**默认区分大小写**
- 从 iOS/macOS 导入的照片文件名大小写可能不一致
- Windows（NTFS）和 macOS（HFS+/APFS 默认配置）不区分大小写，
  使得 stem 比较在这些平台上"碰巧"成功
- 但核心问题是**算法本身缺少大小写规范化**，应修复算法而非添加平台判断

### 建议修复方向 / Suggested Fix

1. **在 stem 比较时统一转换为小写：**
   ```python
   # pairing.py — 修复第二阶段匹配
   stem = Path(photo["rel"]).stem.lower()
   candidates = [v for v in videos.values()
                 if Path(v["rel"]).stem.lower() == stem]
   ```

2. **在目录匹配时也统一路径格式：**
   ```python
   folder = Path(photo["rel"]).parent.as_posix().lower()
   candidates = [v for v in videos.values()
                 if Path(v["rel"]).parent.as_posix().lower() == folder]
   ```

3. **注意：这是算法修复，不是平台适配。**
   修复应在 `core/pairing.py` 中进行，不需要引入任何平台判断代码。
   参见 [跨平台架构约定](./cross-platform-architecture.md) 中"核心原则第3条"。

---

## Bug #3: 地图聚类返回键显示为黑色实心方块

### 现象 / Symptoms

- 从地图聚类进入 Gallery View 时，左上角的返回键应该显示为 `←` 箭头图标
- 在 Linux 下，该图标显示为一个**黑色实心方块**
- Windows 下显示正常

### 涉及文件 / Affected Files

| 文件 | 行号 | 职责 |
|---|---|---|
| `src/iPhoto/gui/ui/widgets/gallery_page.py` | 33-38 | 返回按钮的创建与图标设置 |
| `src/iPhoto/gui/ui/icons.py` | 28-129 | SVG 图标加载与渲染 |
| `src/iPhoto/gui/ui/icon/chevron.left.svg` | — | 返回箭头 SVG 源文件 |

### 根因分析 / Root Cause Analysis

#### 因素 1: `load_icon()` 调用时未指定尺寸

```python
# gallery_page.py:35
self.back_button.setIcon(load_icon("chevron.left.svg"))  # 未传入 size 参数
```

#### 因素 2: SVG 默认尺寸依赖与回退行为

在 `icons.py` 的 `load_icon()` 函数中：

```python
# icons.py:100-102
target_size = QSize(*size) if size else renderer.defaultSize()
if not target_size.isValid():
    target_size = QSize(64, 64)  # 回退到 64×64
```

当未指定 `size` 参数时，函数依赖 `QSvgRenderer.defaultSize()` 从 SVG 文件中
读取默认尺寸。

#### 因素 3: SVG 文件分析

查看 `chevron.left.svg` 的源代码：

```xml
<svg width='29.583984375px' height='34.50390625px' direction='ltr' ...>
```

SVG 文件中使用了 `px` 单位声明了宽高（`29.58 × 34.50`），但没有 `viewBox` 属性。

**平台差异:**
- **Windows:** Qt 的 SVG 渲染器能正确解析带 `px` 单位的 `width`/`height` 属性，
  `defaultSize()` 返回有效尺寸
- **Linux:** Qt 的 SVG 解析器（基于不同的底层库版本）可能在处理带小数点的 `px` 
  尺寸时返回无效的 `QSize()`，从而触发 64×64 回退

#### 因素 4: 快速路径跳过了 SVG 渲染管线

关键发现 — 在 `icons.py` 中有一条快速路径：

```python
# icons.py:84-93
if (
    svg_data is None
    and stroke_width is None
    and normalized_color is None
    and size is None
    and not mirror_horizontal
):
    icon = QIcon(str(path))  # 直接从文件路径创建 QIcon
    _ICON_CACHE[cache_key] = icon
    return icon
```

当 `load_icon("chevron.left.svg")` 不传任何可选参数时，会命中这条快速路径，
直接使用 `QIcon(str(path))` 从文件路径创建图标。

**这意味着:**
- 图标的渲染完全由 Qt 的 SVG 图标引擎在绘制时处理
- Qt 的图标引擎需要根据按钮的实际尺寸和设备像素比动态渲染 SVG
- **Windows:** Qt 的 SVG 图标引擎能正确渲染此 SVG 文件
- **Linux:** Qt 的 SVG 图标引擎在渲染没有 `viewBox` 的 SVG 时可能产生黑色或空白输出

#### 因素 5: SVG 填充颜色为黑色

```xml
<path fill='black' stroke='black' ... />
```

SVG 中的路径使用了 `fill='black'` 和 `stroke='black'`。
在深色主题下，如果图标渲染成功但未进行颜色着色，
黑色图标在深色背景上几乎不可见。
如果 SVG 渲染不完全（仅渲染了填充区域的一部分），看起来就是一个黑色方块。

### 为何仅 Linux 受影响 / Why Linux-Specific

- Qt 在 Linux 和 Windows 上使用的 SVG 渲染后端可能不同（版本、配置）
- `QSvgRenderer.defaultSize()` 对带小数 `px` 值的解析在不同平台有差异
- 快速路径下 `QIcon(path)` 的 SVG 图标引擎行为在不同平台有差异
- 缺少 `viewBox` 属性使得 SVG 的可缩放性依赖于解析器实现

### 建议修复方向 / Suggested Fix

1. **始终为图标指定明确尺寸（推荐）：**
   ```python
   # gallery_page.py:35 — 修复
   self.back_button.setIcon(load_icon("chevron.left.svg", size=(16, 16)))
   ```

2. **为 SVG 文件添加 `viewBox` 属性：**
   ```xml
   <svg width='29.583984375px' height='34.50390625px'
        viewBox='0 0 29.583984375 34.50390625' ...>
   ```

3. **在深色主题下为图标指定颜色（可选增强）：**
   ```python
   self.back_button.setIcon(
       load_icon("chevron.left.svg", size=(16, 16), color="#FFFFFF")
   )
   ```

4. **注意：这属于 Qt 渲染差异，应在 GUI 层修复。**
   参见 [跨平台架构约定](./cross-platform-architecture.md) 中"图形渲染适配"部分。

---

## Bug #4: 返回键切换界面后不能正确消失

### 现象 / Symptoms

- 从地图聚类进入 Gallery View，此时返回键正确显示
- **不点击返回键**，而是通过侧边栏切换到其他视图（如 All Photos、某个相册、收藏夹等）
- 切换后返回键仍然显示在 Gallery View 上方
- 点击该残留的返回键会触发错误的导航行为（返回地图而非预期行为）

### 涉及文件 / Affected Files

| 文件 | 行号 | 职责 |
|---|---|---|
| `src/iPhoto/gui/coordinators/navigation_coordinator.py` | 66 | `_in_cluster_gallery` 状态标志 |
| `src/iPhoto/gui/coordinators/navigation_coordinator.py` | 91-122 | `open_album()` — **缺少状态重置** |
| `src/iPhoto/gui/coordinators/navigation_coordinator.py` | 124-132 | `open_all_photos()` — **缺少状态重置** |
| `src/iPhoto/gui/coordinators/navigation_coordinator.py` | 154-186 | `open_recently_deleted()` — **缺少状态重置** |
| `src/iPhoto/gui/coordinators/navigation_coordinator.py` | 188-201 | `_open_filtered_collection()` — **缺少状态重置** |
| `src/iPhoto/gui/coordinators/navigation_coordinator.py` | 134-152 | `_handle_static_node()` — **缺少状态重置** |
| `src/iPhoto/gui/ui/widgets/gallery_page.py` | 50-57 | `set_cluster_gallery_mode()` — 仅被两处调用 |
| `src/iPhoto/gui/coordinators/main_coordinator.py` | 538-558 | `_handle_back_button()` — 依赖状态准确性 |

### 根因分析 / Root Cause Analysis

这是一个**状态管理遗漏**问题，与平台无关，但在 Linux 上更容易被观察到。

#### 状态流转分析

`NavigationCoordinator` 使用 `_in_cluster_gallery: bool` 标志跟踪是否正在查看
地图聚类相册：

```python
# navigation_coordinator.py:66
self._in_cluster_gallery: bool = False
```

**正确设置该状态的方法：**

| 方法 | 设置为 | 行号 |
|---|---|---|
| `open_cluster_gallery()` | `True` | 247 |
| `return_to_map_from_cluster_gallery()` | `False` | 270 |
| `open_location_view()` | `False` | 213 |

**缺少状态重置的方法：**

| 方法 | 行号 | 缺少的操作 |
|---|---|---|
| `open_album()` | 91-122 | 未重置 `_in_cluster_gallery = False` |
| `open_all_photos()` | 124-132 | 未重置 `_in_cluster_gallery = False` |
| `open_recently_deleted()` | 154-186 | 未重置 `_in_cluster_gallery = False` |
| `_open_filtered_collection()` | 188-201 | 未重置 `_in_cluster_gallery = False` |
| `_handle_static_node()` | 134-152 | 未调用状态重置（Albums Dashboard 分支） |

#### 同时遗漏 UI 状态同步

不仅 `_in_cluster_gallery` 标志未被重置，`gallery_page.set_cluster_gallery_mode(False)`
也从未在这些方法中被调用：

```python
# 目前 set_cluster_gallery_mode 仅在以下两处被调用:
# 1. open_cluster_gallery()   → set_cluster_gallery_mode(True)   ✅
# 2. return_to_map_from_cluster_gallery() → set_cluster_gallery_mode(False) ✅
# 
# 以下方法都切换到 gallery view 但未调用 set_cluster_gallery_mode(False):
# - open_album()                    ❌
# - open_all_photos()               ❌
# - open_recently_deleted()         ❌
# - _open_filtered_collection()     ❌
```

#### 完整的 Bug 复现序列

```
1. 用户点击侧边栏 "Location" → open_location_view()
   → _in_cluster_gallery = False ✅

2. 用户在地图上点击聚类 → open_cluster_gallery(assets)
   → _in_cluster_gallery = True ✅
   → gallery_page.set_cluster_gallery_mode(True) ✅
   → 返回键可见 ✅

3. 用户点击侧边栏 "All Photos" → open_all_photos()
   → 切换到 Gallery View
   → _in_cluster_gallery 仍为 True ❌ (未重置)
   → gallery_page header 仍然可见 ❌ (未调用 set_cluster_gallery_mode(False))
   → 返回键残留 ❌

4. 如果用户点击残留的返回键:
   → main_coordinator._handle_back_button()
   → is_in_cluster_gallery() == True (错误!)
   → is_gallery_view_active() == True
   → 调用 return_to_map_from_cluster_gallery()
   → 错误地跳转回地图视图 ❌
```

### 为何 Linux 更易暴露 / Why More Visible on Linux

- **此 Bug 本身是跨平台的**，在任何平台上都会出现
- 在 Linux 上更容易被注意到的原因可能是：
  - Bug #3（返回键为黑色方块）使得该控件更加醒目
  - Linux 的事件处理时序可能使得 widget 的 `setVisible` 状态更新更加即时
  - 测试人员在 Linux 上更仔细地测试了导航流程

### 建议修复方向 / Suggested Fix

**核心修复：** 创建一个统一的状态清理方法，在所有切换视图的导航方法中调用。

```python
# navigation_coordinator.py — 新增方法
def _exit_cluster_gallery_mode(self) -> None:
    """Reset cluster gallery state when navigating away."""
    if not self._in_cluster_gallery:
        return
    self._in_cluster_gallery = False
    gallery_page = self._router.gallery_page()
    if gallery_page is not None:
        gallery_page.set_cluster_gallery_mode(False)
```

然后在所有导航方法中调用：

```python
def open_album(self, path: Path):
    ...
    self._exit_cluster_gallery_mode()  # 新增
    self._reset_playback()
    ...

def open_all_photos(self):
    ...
    self._exit_cluster_gallery_mode()  # 新增
    self._reset_playback()
    ...

def open_recently_deleted(self):
    ...
    self._exit_cluster_gallery_mode()  # 新增
    self._reset_playback()
    ...

def _open_filtered_collection(self, title, ...):
    ...
    self._exit_cluster_gallery_mode()  # 新增
    self._reset_playback()
    ...

def _handle_static_node(self, name: str):
    ...
    elif normalized == "albums":
        self._exit_cluster_gallery_mode()  # 新增
        self._reset_playback()
        ...
```

**注意：这是逻辑修复，不是平台适配。**
参见 [跨平台架构约定](./cross-platform-architecture.md) 中"核心原则第3条"——
修复算法本身使其在所有平台通用。

---

## 总结与优先级 / Summary & Priority

| Bug | 根因分类 | 是否跨平台 | 修复层级 | 优先级 |
|---|---|---|---|---|
| **#1** 首张缩略图空白 | Qt/OpenGL 渲染时序 | Linux 特有 | `gui/widgets/` + 可选 `platform/graphics.py` | 🟡 中 |
| **#2** Live Photo 配对不正确 | 算法缺陷（大小写敏感性） | 所有平台（Linux 更易暴露） | `core/pairing.py` | 🔴 高 |
| **#3** 返回键为黑色方块 | SVG 渲染 + 未指定尺寸 | Linux 特有 | `gui/widgets/gallery_page.py` | 🟡 中 |
| **#4** 返回键不消失 | 状态管理遗漏 | 所有平台 | `gui/coordinators/navigation_coordinator.py` | 🔴 严重 |

### 修复原则映射 / Fix Principle Mapping

| Bug | 修复原则 |
|---|---|
| **#1** | 参照架构约定 → 渲染差异应在 `platform/graphics.py` 或 GUI 层处理 |
| **#2** | 参照架构约定 → **修复算法本身**，不引入平台判断 |
| **#3** | 参照架构约定 → GUI 层修复（指定图标尺寸），不需要平台判断 |
| **#4** | 参照架构约定 → **修复算法本身**，不引入平台判断 |

### 关键结论 / Key Takeaways

1. **4 个 Bug 中只有 1 个真正需要平台特定代码**（Bug #1 的 OpenGL 初始化时序）
2. **Bug #2 和 #4 是算法/逻辑 Bug**，在所有平台都存在，应修复算法本身
3. **Bug #3 可以通过更规范的 API 调用来修复**（明确指定图标尺寸），无需平台判断
4. 这印证了[跨平台架构约定](./cross-platform-architecture.md)中的核心原则：
   **大多数"平台差异"实际上是代码本身的健壮性不足**

---

## 相关文档 / Related Documents

- [跨平台架构约定 / Cross-Platform Architecture Conventions](./cross-platform-architecture.md)
- [架构文档 / Architecture](./architecture.md)
- [开发指南 / Development](./development.md)
