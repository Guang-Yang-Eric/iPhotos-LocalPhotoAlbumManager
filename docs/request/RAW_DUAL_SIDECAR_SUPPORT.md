# RAW 文件支持与双侧车（Dual Sidecar）开发文档

## 1. 概述

本功能为 iPhotos 相册管理器加入 RAW 文件支持，并实现双侧车（Dual Sidecar）机制：
- **`.ipo` 文件**（iPhoto 原生格式）——默认读写格式
- **Adobe `.xmp` 文件**——仅在用户手动导出时生成，支持从 XMP 导入

两个格式之间可互相转化，且转化过程保证 **结果一致性**（而非过程一致性）。

## 2. 设计原则

### 2.1 结果一致性 vs 过程一致性

传统做法是在侧车文件中存储"曲线控制点"或"参数"，但不同软件（如 iPhotos、Adobe Lightroom、Capture One）对同一组控制点的解析算法不同，导致最终渲染颜色不一致。

**本方案采用"已烘焙的颜色变换结果"存储策略：**

| 调整类型 | 存储方式 | 说明 |
|---------|---------|------|
| 曲线（Curves） | 256×3 LUT（查找表） | 存储最终的 R/G/B 映射结果而非控制点 |
| 色阶（Levels） | 256×3 LUT | 与曲线合并为同一 LUT |
| 亮度/对比度等 | 标量参数 | 语义明确，跨软件一致 |
| 裁剪/旋转 | 归一化坐标 | 数学定义明确，无歧义 |
| 白平衡 | 温度/色调标量 | 可安全跨格式映射 |
| 选择性颜色 | 色相范围 + 偏移 | 语义参数，跨软件一致 |
| 黑白 | 强度/色调参数 | 标量值，语义明确 |

### 2.2 LUT 存储格式

在 `.ipo` 文件中，LUT 以 Base64 编码的二进制数据存储：

```xml
<BakedLUT enabled="true" size="256">
    <!-- Base64 编码的 256×3 float32 数组 (3072 字节) -->
    <data>AAAA/wAA...</data>
</BakedLUT>
```

在 `.xmp` 文件中，LUT 通过 Adobe XMP 的 `crs:ToneCurvePV2012` 序列表示，或者作为自定义命名空间中的 Base64 数据存储。

### 2.3 侧车文件优先级

```
读取流程:
1. 检查 .ipo 文件是否存在 → 存在则读取
2. .ipo 不存在时，检查 .xmp 文件 → 存在则读取并自动转换为内部格式

保存流程:
1. 始终保存为 .ipo 文件（自动）
2. .xmp 文件仅在用户手动导出时生成
```

## 3. 支持的 RAW 格式

| 扩展名 | 相机品牌 |
|--------|---------|
| `.cr2`, `.cr3` | Canon |
| `.nef`, `.nrw` | Nikon |
| `.arw`, `.srf`, `.sr2` | Sony |
| `.orf` | Olympus |
| `.rw2` | Panasonic |
| `.raf` | Fujifilm |
| `.dng` | Adobe DNG（通用） |
| `.pef` | Pentax |
| `.raw`, `.rwl` | Leica |
| `.3fr` | Hasselblad |
| `.iiq` | Phase One |
| `.x3f` | Sigma |
| `.srw` | Samsung |
| `.erf` | Epson |

## 4. 模块结构

### 4.1 文件变更列表

```
src/iPhoto/
├── media_classifier.py         # 新增 RAW_EXTENSIONS
├── config.py                   # DEFAULT_INCLUDE 新增 RAW 扩展名
├── io/
│   ├── sidecar.py              # 更新：双侧车优先级逻辑 + LUT 存储
│   └── xmp_sidecar.py          # 新增：XMP 读写 + IPO↔XMP 转换
└── core/
    └── (existing resolvers)    # 无需修改，已有 LUT 生成能力
```

### 4.2 核心接口

#### `sidecar.py` 更新

```python
# 新增：支持 RAW 文件的侧车路径解析
def sidecar_path_for_asset(asset_path: Path) -> Path:
    """返回 .ipo 侧车路径（优先）"""

# 更新：双侧车加载
def load_adjustments(asset_path: Path) -> Dict[str, Any]:
    """
    加载顺序：
    1. 检查 .ipo 文件
    2. .ipo 不存在时检查 .xmp 文件
    """

# 更新：始终保存为 .ipo
def save_adjustments(asset_path: Path, adjustments: Mapping[str, Any]) -> Path:
    """始终保存为 .ipo 文件"""
```

#### `xmp_sidecar.py` 新增

```python
def xmp_sidecar_path_for_asset(asset_path: Path) -> Path:
    """返回 .xmp 侧车路径"""

def load_xmp_adjustments(asset_path: Path) -> Dict[str, Any]:
    """从 XMP 文件读取调整参数并转换为内部格式"""

def export_xmp(asset_path: Path, adjustments: Mapping[str, Any]) -> Path:
    """将内部调整参数导出为 Adobe XMP 格式"""

def ipo_to_xmp(adjustments: Dict[str, Any]) -> str:
    """将 IPO 内部格式转换为 XMP XML 字符串"""

def xmp_to_ipo(xmp_content: str) -> Dict[str, Any]:
    """将 XMP XML 字符串解析为 IPO 内部格式"""
```

## 5. XMP 兼容性方案

### 5.1 XMP 命名空间

```xml
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
        xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
        xmlns:ipo="http://ns.iphoto.app/sidecar/1.0/">
      
      <!-- Adobe Camera Raw 标准参数 -->
      <crs:Exposure2012>+0.50</crs:Exposure2012>
      <crs:Contrast2012>+10</crs:Contrast2012>
      <crs:Highlights2012>-20</crs:Highlights2012>
      <crs:Shadows2012>+30</crs:Shadows2012>
      
      <!-- 已烘焙的 LUT（iPhoto 自定义命名空间） -->
      <ipo:BakedLUT>base64_encoded_data</ipo:BakedLUT>
      
      <!-- 裁剪信息 -->
      <crs:CropTop>0.1</crs:CropTop>
      <crs:CropLeft>0.2</crs:CropLeft>
      <crs:CropBottom>0.9</crs:CropBottom>
      <crs:CropRight>0.8</crs:CropRight>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

### 5.2 参数映射表

| iPhoto 内部参数 | XMP/CRS 参数 | 转换公式 |
|----------------|-------------|---------|
| `Exposure` | `crs:Exposure2012` | `xmp = ipo × 5.0`（iPhoto [-1,1] → XMP [-5,+5]） |
| `Contrast` | `crs:Contrast2012` | `xmp = ipo × 100`（iPhoto [-1,1] → XMP [-100,+100]） |
| `Highlights` | `crs:Highlights2012` | `xmp = ipo × 100` |
| `Shadows` | `crs:Shadows2012` | `xmp = ipo × 100` |
| `Brightness` | `crs:Brightness` | `xmp = ipo × 150`（iPhoto [-1,1] → XMP [-150,+150]） |
| `BlackPoint` | `crs:Blacks2012` | `xmp = ipo × 100` |
| `Saturation` | `crs:Saturation` | `xmp = ipo × 100` |
| `Vibrance` | `crs:Vibrance` | `xmp = ipo × 100` |
| `WB_Temperature` | `crs:Temperature` | `xmp = 5500 + ipo × 4500`（映射到色温 K） |
| `WB_Tint` | `crs:Tint` | `xmp = ipo × 150` |
| `Crop_*` | `crs:Crop*` | 坐标系转换（中心 ↔ 边界） |

### 5.3 LUT 烘焙策略

当导出为 XMP 时，对于无法精确用 CRS 参数表达的效果（如自定义曲线），采用以下策略：

1. **生成 256×3 LUT**：调用现有的 `generate_curve_lut()` 和 `build_levels_lut()`
2. **存储为 iPhoto 自定义命名空间**：`ipo:BakedLUT`
3. **同时尝试近似映射 CRS 参数**：为 Adobe 软件提供最佳近似值

当从 XMP 导入时：
1. 优先读取 `ipo:BakedLUT`（如果存在）
2. 否则从 CRS 参数重建调整值

## 6. 数据流图

```
                    ┌─────────────┐
                    │   RAW 文件    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  文件扫描器   │ ← media_classifier.py
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      ┌───────────┐  ┌───────────┐  ┌───────────┐
      │ .ipo 存在  │  │ .xmp 存在  │  │  均不存在   │
      │ (优先读取)  │  │ (备选读取)  │  │ (空调整集)  │
      └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
            │              │              │
            └──────┬───────┘              │
                   ▼                      │
           ┌─────────────┐               │
           │ 内部调整格式  │◄──────────────┘
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │   编辑会话   │
           └──────┬──────┘
                  │
         ┌────────┼────────┐
         ▼                 ▼
   ┌───────────┐    ┌───────────┐
   │ 自动保存    │    │ 手动导出   │
   │ → .ipo    │    │ → .xmp   │
   └───────────┘    └───────────┘
```

## 7. 测试计划

| 测试用例 | 描述 |
|---------|------|
| RAW 格式识别 | 验证所有 RAW 扩展名被正确分类为 IMAGE |
| 双侧车优先级 | .ipo 和 .xmp 同时存在时优先读取 .ipo |
| XMP 导入 | 从标准 XMP 文件正确解析调整参数 |
| XMP 导出 | 将内部参数正确映射为 XMP 格式 |
| IPO↔XMP 互转 | 双向转换后参数一致性验证 |
| LUT 烘焙 | 曲线/色阶生成的 LUT 在转换后保持一致 |
| 向后兼容 | 现有 .ipo 文件无需修改即可正常加载 |

## 8. 版本兼容性

- `.ipo` 文件版本保持 `1.0`，新增的 `<BakedLUT>` 节点为可选
- 旧版本 iPhotos 可安全忽略未知 XML 节点
- XMP 文件遵循 Adobe XMP 规范，兼容 Lightroom/ACR
