# 使用 pybind11 替换 numba 的详细开发文档

## 1. 概述

本文档详细说明如何使用 pybind11 和 C++ 替换项目中当前所有使用 numba 加速的 Python 算法。此迁移旨在提高性能、减少依赖、并为未来的优化提供更大的灵活性。

### 1.1 迁移目标

- 使用 C++17/C++20 实现高性能图像处理算法
- 通过 pybind11 提供 Python 绑定
- 保持与现有 Python API 的完全兼容性
- 提供比 numba JIT 更快的编译时优化代码
- 减少运行时依赖（不再需要 numba 和 LLVM）

### 1.2 当前 numba 使用情况

项目中使用 numba 的文件列表：

1. `src/iPhoto/core/color_resolver.py` - 颜色统计计算中的辅助函数
2. `src/iPhoto/core/filters/algorithms.py` - 核心图像处理算法
3. `src/iPhoto/core/filters/jit_kernels.py` - JIT 编译的图像处理内核
4. `src/iPhoto/core/filters/jit_executor.py` - JIT 执行器
5. `src/iPhoto/core/filters/build_jit.py` - AOT 编译脚本

## 2. 新文件结构

### 2.1 C++ 源代码目录结构

```
iPhotos-LocalPhotoAlbumManager/
├── cpp/
│   ├── include/
│   │   ├── iphoto/
│   │   │   ├── algorithms.hpp        # 算法声明
│   │   │   ├── color_algorithms.hpp  # 颜色处理算法
│   │   │   ├── image_kernels.hpp     # 图像处理内核
│   │   │   ├── math_utils.hpp        # 数学工具函数
│   │   │   └── types.hpp             # 公共类型定义
│   ├── src/
│   │   ├── algorithms.cpp            # 通道调整算法实现
│   │   ├── color_algorithms.cpp      # 颜色变换实现
│   │   ├── image_kernels.cpp         # 主图像处理内核
│   │   ├── bw_algorithms.cpp         # 黑白效果算法
│   │   ├── math_utils.cpp            # 数学工具实现
│   │   └── bindings.cpp              # pybind11 Python 绑定
│   ├── tests/
│   │   ├── test_algorithms.cpp       # C++ 单元测试
│   │   ├── test_kernels.cpp
│   │   └── CMakeLists.txt
│   ├── CMakeLists.txt                # 主 CMake 配置
│   └── README.md                     # C++ 模块说明
├── src/iPhoto/core/filters/
│   ├── _cpp_filters.so              # 编译后的 C++ 扩展（Linux）
│   ├── _cpp_filters.pyd             # 编译后的 C++ 扩展（Windows）
│   ├── cpp_executor.py              # C++ 执行器包装器
│   └── ... (保留现有文件作为备份/对比)
└── pyproject.toml                    # 更新构建配置
```

### 2.2 需要创建的 C++ 头文件

#### 2.2.1 `cpp/include/iphoto/types.hpp`
```cpp
#pragma once

#include <cstdint>
#include <array>

namespace iphoto {

// 基础类型定义
using u8 = std::uint8_t;
using f32 = float;
using f64 = double;

// RGB 颜色结构
struct RGB {
    f32 r, g, b;
};

// 图像缓冲区信息
struct ImageBuffer {
    u8* data;
    int width;
    int height;
    int bytes_per_line;
};

} // namespace iphoto
```

#### 2.2.2 `cpp/include/iphoto/math_utils.hpp`
```cpp
#pragma once

#include "types.hpp"
#include <algorithm>
#include <cmath>

namespace iphoto {
namespace math {

// 对应 Python 中的 _clamp 函数
inline f32 clamp(f32 value, f32 min_val, f32 max_val) noexcept {
    return std::clamp(value, min_val, max_val);
}

inline f32 clamp01(f32 value) noexcept {
    return clamp(value, 0.0f, 1.0f);
}

// 对应 Python 中的 _mix 函数
inline f32 mix(f32 a, f32 b, f32 t) noexcept {
    t = clamp01(t);
    return a * (1.0f - t) + b * t;
}

// 快速整数转换
inline u8 float_to_uint8(f32 value) noexcept {
    auto scaled = static_cast<int>(std::round(value * 255.0f));
    return static_cast<u8>(std::clamp(scaled, 0, 255));
}

} // namespace math
} // namespace iphoto
```

#### 2.2.3 `cpp/include/iphoto/algorithms.hpp`
```cpp
#pragma once

#include "types.hpp"

namespace iphoto {
namespace algorithms {

// 对应 _apply_channel_adjustments
f32 apply_channel_adjustments(
    f32 value,
    f32 exposure,
    f32 brightness,
    f32 brilliance,
    f32 highlights,
    f32 shadows,
    f32 contrast_factor,
    f32 black_point
) noexcept;

// 对应 _apply_color_transform
RGB apply_color_transform(
    f32 r, f32 g, f32 b,
    f32 saturation,
    f32 vibrance,
    f32 cast,
    f32 gain_r,
    f32 gain_g,
    f32 gain_b
) noexcept;

// 对应 _apply_bw_channels
RGB apply_bw_channels(
    f32 r, f32 g, f32 b,
    f32 intensity,
    f32 neutrals,
    f32 tone,
    f32 grain,
    f32 noise
) noexcept;

// 对应 _grain_noise
f32 grain_noise(int x, int y, int width, int height) noexcept;

// 辅助函数
f32 gamma_neutral(f32 value, f32 neutrals) noexcept;
f32 contrast_tone_curve(f32 value, f32 tone) noexcept;

} // namespace algorithms
} // namespace iphoto
```

#### 2.2.4 `cpp/include/iphoto/image_kernels.hpp`
```cpp
#pragma once

#include "types.hpp"

namespace iphoto {
namespace kernels {

// 对应 _apply_adjustments_fast
void apply_adjustments_fast(
    u8* buffer,
    int width,
    int height,
    int bytes_per_line,
    f32 exposure_term,
    f32 brightness_term,
    f32 brilliance_strength,
    f32 highlights,
    f32 shadows,
    f32 contrast_factor,
    f32 black_point,
    f32 saturation,
    f32 vibrance,
    f32 cast,
    f32 gain_r,
    f32 gain_g,
    f32 gain_b,
    bool apply_color,
    bool apply_bw,
    f32 bw_intensity,
    f32 bw_neutrals,
    f32 bw_tone,
    f32 bw_grain
) noexcept;

// 对应 _apply_color_adjustments_inplace
void apply_color_adjustments_inplace(
    u8* buffer,
    int width,
    int height,
    int bytes_per_line,
    f32 saturation,
    f32 vibrance,
    f32 cast,
    f32 gain_r,
    f32 gain_g,
    f32 gain_b
) noexcept;

} // namespace kernels
} // namespace iphoto
```

### 2.3 需要创建的 C++ 实现文件

#### 2.3.1 `cpp/src/algorithms.cpp`

关键实现示例：

```cpp
#include "iphoto/algorithms.hpp"
#include "iphoto/math_utils.hpp"

namespace iphoto {
namespace algorithms {

f32 apply_channel_adjustments(
    f32 value,
    f32 exposure,
    f32 brightness,
    f32 brilliance,
    f32 highlights,
    f32 shadows,
    f32 contrast_factor,
    f32 black_point
) noexcept {
    using namespace math;
    
    // Exposure/brightness adjustments
    f32 adjusted = value + exposure + brightness;
    
    // Brilliance - affects mid-tones
    f32 mid_distance = value - 0.5f;
    adjusted += brilliance * (1.0f - std::pow(mid_distance * 2.0f, 2.0f));
    
    // Highlights and shadows
    if (adjusted > 0.65f) {
        f32 ratio = (adjusted - 0.65f) / 0.35f;
        adjusted += highlights * ratio;
    } else if (adjusted < 0.35f) {
        f32 ratio = (0.35f - adjusted) / 0.35f;
        adjusted += shadows * ratio;
    }
    
    // Contrast
    adjusted = (adjusted - 0.5f) * contrast_factor + 0.5f;
    
    // Black point
    if (black_point > 0.0f) {
        adjusted -= black_point * (1.0f - adjusted);
    } else if (black_point < 0.0f) {
        adjusted -= black_point * adjusted;
    }
    
    return clamp01(adjusted);
}

RGB apply_color_transform(
    f32 r, f32 g, f32 b,
    f32 saturation,
    f32 vibrance,
    f32 cast,
    f32 gain_r,
    f32 gain_g,
    f32 gain_b
) noexcept {
    using namespace math;
    
    // White balance / cast correction
    f32 mix_r = (1.0f - cast) + gain_r * cast;
    f32 mix_g = (1.0f - cast) + gain_g * cast;
    f32 mix_b = (1.0f - cast) + gain_b * cast;
    
    r *= mix_r;
    g *= mix_g;
    b *= mix_b;
    
    // Luma-chroma decomposition
    f32 luma = 0.299f * r + 0.587f * g + 0.114f * b;
    f32 chroma_r = r - luma;
    f32 chroma_g = g - luma;
    f32 chroma_b = b - luma;
    
    // Saturation and vibrance
    f32 sat_amt = 1.0f + saturation;
    f32 vib_amt = 1.0f + vibrance;
    f32 w = 1.0f - clamp(std::abs(luma - 0.5f) * 2.0f, 0.0f, 1.0f);
    f32 chroma_scale = sat_amt * (1.0f + (vib_amt - 1.0f) * w);
    
    chroma_r *= chroma_scale;
    chroma_g *= chroma_scale;
    chroma_b *= chroma_scale;
    
    return RGB{
        clamp(luma + chroma_r, 0.0f, 1.0f),
        clamp(luma + chroma_g, 0.0f, 1.0f),
        clamp(luma + chroma_b, 0.0f, 1.0f)
    };
}

f32 grain_noise(int x, int y, int width, int height) noexcept {
    using namespace math;
    
    if (width <= 0 || height <= 0) {
        return 0.5f;
    }
    
    f32 u = static_cast<f32>(x) / static_cast<f32>(std::max(width - 1, 1));
    f32 v = static_cast<f32>(y) / static_cast<f32>(std::max(height - 1, 1));
    
    // Sine-based pseudo-random hash (matching shader logic)
    f32 seed = u * 12.9898f + v * 78.233f;
    f32 noise = std::sin(seed) * 43758.5453f;
    f32 fraction = noise - std::floor(noise);
    
    return clamp01(fraction);
}

f32 gamma_neutral(f32 value, f32 neutrals) noexcept {
    using namespace math;
    
    neutrals = clamp01(neutrals);
    f32 n = 0.6f * (neutrals - 0.5f);
    f32 gamma = std::pow(2.0f, -n * 2.0f);
    
    return clamp(std::pow(clamp(value, 0.0f, 1.0f), gamma), 0.0f, 1.0f);
}

f32 contrast_tone_curve(f32 value, f32 tone) noexcept {
    using namespace math;
    
    tone = clamp01(tone);
    f32 t = tone - 0.5f;
    f32 factor = (t >= 0.0f) ? mix(1.0f, 2.2f, t * 2.0f) 
                              : mix(1.0f, 0.6f, -t * 2.0f);
    
    f32 x = clamp(value, 0.0f, 1.0f);
    constexpr f32 eps = 1e-6f;
    f32 pos = clamp(x, eps, 1.0f - eps);
    f32 logit = std::log(pos / std::max(eps, 1.0f - pos));
    f32 y = 1.0f / (1.0f + std::exp(-logit * factor));
    
    return clamp(y, 0.0f, 1.0f);
}

RGB apply_bw_channels(
    f32 r, f32 g, f32 b,
    f32 intensity,
    f32 neutrals,
    f32 tone,
    f32 grain,
    f32 noise
) noexcept {
    using namespace math;
    
    intensity = clamp01(intensity);
    neutrals = clamp01(neutrals);
    tone = clamp01(tone);
    grain = clamp01(grain);
    noise = clamp01(noise);
    
    // Compute luma
    f32 luma = clamp(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f, 1.0f);
    
    // Different intensity modes
    f32 soft_base = clamp(std::pow(luma, 0.82f), 0.0f, 1.0f);
    f32 soft_curve = contrast_tone_curve(soft_base, 0.0f);
    f32 g_soft = (soft_curve + soft_base) * 0.5f;
    f32 g_neutral = luma;
    f32 g_rich = contrast_tone_curve(
        clamp(std::pow(luma, 1.0f / 1.22f), 0.0f, 1.0f), 
        0.35f
    );
    
    f32 gray;
    if (intensity >= 0.5f) {
        f32 blend = (intensity - 0.5f) / 0.5f;
        gray = mix(g_neutral, g_rich, blend);
    } else {
        f32 blend = (0.5f - intensity) / 0.5f;
        gray = mix(g_soft, g_neutral, blend);
    }
    
    // Apply neutrals and tone
    gray = gamma_neutral(gray, neutrals);
    gray = contrast_tone_curve(gray, tone);
    
    // Add grain
    if (grain > 1e-6f) {
        gray += (noise - 0.5f) * 0.2f * grain;
    }
    
    f32 clamped = clamp01(gray);
    return RGB{clamped, clamped, clamped};
}

} // namespace algorithms
} // namespace iphoto
```

#### 2.3.2 `cpp/src/image_kernels.cpp`

```cpp
#include "iphoto/image_kernels.hpp"
#include "iphoto/algorithms.hpp"
#include "iphoto/math_utils.hpp"

namespace iphoto {
namespace kernels {

void apply_adjustments_fast(
    u8* buffer,
    int width,
    int height,
    int bytes_per_line,
    f32 exposure_term,
    f32 brightness_term,
    f32 brilliance_strength,
    f32 highlights,
    f32 shadows,
    f32 contrast_factor,
    f32 black_point,
    f32 saturation,
    f32 vibrance,
    f32 cast,
    f32 gain_r,
    f32 gain_g,
    f32 gain_b,
    bool apply_color,
    bool apply_bw,
    f32 bw_intensity,
    f32 bw_neutrals,
    f32 bw_tone,
    f32 bw_grain
) noexcept {
    using namespace algorithms;
    using namespace math;
    
    if (width <= 0 || height <= 0) {
        return;
    }
    
    // Process each pixel
    for (int y = 0; y < height; ++y) {
        size_t row_offset = static_cast<size_t>(y) * bytes_per_line;
        
        for (int x = 0; x < width; ++x) {
            size_t pixel_offset = row_offset + static_cast<size_t>(x) * 4;
            
            // Read pixel (BGRA format)
            f32 b = buffer[pixel_offset] / 255.0f;
            f32 g = buffer[pixel_offset + 1] / 255.0f;
            f32 r = buffer[pixel_offset + 2] / 255.0f;
            // Alpha at pixel_offset + 3 is preserved
            
            // Apply channel adjustments
            r = apply_channel_adjustments(
                r, exposure_term, brightness_term, brilliance_strength,
                highlights, shadows, contrast_factor, black_point
            );
            g = apply_channel_adjustments(
                g, exposure_term, brightness_term, brilliance_strength,
                highlights, shadows, contrast_factor, black_point
            );
            b = apply_channel_adjustments(
                b, exposure_term, brightness_term, brilliance_strength,
                highlights, shadows, contrast_factor, black_point
            );
            
            // Apply color transform if needed
            if (apply_color) {
                auto rgb = apply_color_transform(
                    r, g, b, saturation, vibrance, cast,
                    gain_r, gain_g, gain_b
                );
                r = rgb.r;
                g = rgb.g;
                b = rgb.b;
            }
            
            // Apply black & white effect if needed
            if (apply_bw) {
                f32 noise = 0.0f;
                if (std::abs(bw_grain) > 1e-6f) {
                    noise = grain_noise(x, y, width, height);
                }
                
                auto rgb = apply_bw_channels(
                    r, g, b, bw_intensity, bw_neutrals,
                    bw_tone, bw_grain, noise
                );
                r = rgb.r;
                g = rgb.g;
                b = rgb.b;
            }
            
            // Write back to buffer (BGRA format)
            buffer[pixel_offset] = float_to_uint8(b);
            buffer[pixel_offset + 1] = float_to_uint8(g);
            buffer[pixel_offset + 2] = float_to_uint8(r);
            // Alpha channel unchanged
        }
    }
}

void apply_color_adjustments_inplace(
    u8* buffer,
    int width,
    int height,
    int bytes_per_line,
    f32 saturation,
    f32 vibrance,
    f32 cast,
    f32 gain_r,
    f32 gain_g,
    f32 gain_b
) noexcept {
    using namespace algorithms;
    using namespace math;
    
    if (width <= 0 || height <= 0) {
        return;
    }
    
    bool apply_color = std::abs(saturation) > 1e-6f || 
                       std::abs(vibrance) > 1e-6f || 
                       cast > 1e-6f;
    
    if (!apply_color) {
        return;
    }
    
    for (int y = 0; y < height; ++y) {
        size_t row_offset = static_cast<size_t>(y) * bytes_per_line;
        
        for (int x = 0; x < width; ++x) {
            size_t pixel_offset = row_offset + static_cast<size_t>(x) * 4;
            
            f32 b = buffer[pixel_offset] / 255.0f;
            f32 g = buffer[pixel_offset + 1] / 255.0f;
            f32 r = buffer[pixel_offset + 2] / 255.0f;
            
            auto rgb = apply_color_transform(
                r, g, b, saturation, vibrance, cast,
                gain_r, gain_g, gain_b
            );
            
            buffer[pixel_offset] = float_to_uint8(rgb.b);
            buffer[pixel_offset + 1] = float_to_uint8(rgb.g);
            buffer[pixel_offset + 2] = float_to_uint8(rgb.r);
        }
    }
}

} // namespace kernels
} // namespace iphoto
```

#### 2.3.3 `cpp/src/bindings.cpp` - pybind11 绑定

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "iphoto/image_kernels.hpp"
#include "iphoto/algorithms.hpp"
#include "iphoto/math_utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_filters, m) {
    m.doc() = "C++ accelerated image processing filters for iPhoto";
    
    // Expose main kernel functions
    m.def("apply_adjustments_fast",
        [](py::array_t<uint8_t> buffer,
           int width,
           int height,
           int bytes_per_line,
           float exposure_term,
           float brightness_term,
           float brilliance_strength,
           float highlights,
           float shadows,
           float contrast_factor,
           float black_point,
           float saturation,
           float vibrance,
           float cast,
           float gain_r,
           float gain_g,
           float gain_b,
           bool apply_color,
           bool apply_bw,
           float bw_intensity,
           float bw_neutrals,
           float bw_tone,
           float bw_grain) {
            
            auto buf = buffer.mutable_unchecked<1>();
            uint8_t* data = buf.mutable_data(0);
            
            iphoto::kernels::apply_adjustments_fast(
                data, width, height, bytes_per_line,
                exposure_term, brightness_term, brilliance_strength,
                highlights, shadows, contrast_factor, black_point,
                saturation, vibrance, cast,
                gain_r, gain_g, gain_b,
                apply_color, apply_bw,
                bw_intensity, bw_neutrals, bw_tone, bw_grain
            );
        },
        py::arg("buffer"),
        py::arg("width"),
        py::arg("height"),
        py::arg("bytes_per_line"),
        py::arg("exposure_term"),
        py::arg("brightness_term"),
        py::arg("brilliance_strength"),
        py::arg("highlights"),
        py::arg("shadows"),
        py::arg("contrast_factor"),
        py::arg("black_point"),
        py::arg("saturation"),
        py::arg("vibrance"),
        py::arg("cast"),
        py::arg("gain_r"),
        py::arg("gain_g"),
        py::arg("gain_b"),
        py::arg("apply_color"),
        py::arg("apply_bw"),
        py::arg("bw_intensity"),
        py::arg("bw_neutrals"),
        py::arg("bw_tone"),
        py::arg("bw_grain"),
        "Apply full image adjustments pipeline"
    );
    
    m.def("apply_color_adjustments_inplace",
        [](py::array_t<uint8_t> buffer,
           int width,
           int height,
           int bytes_per_line,
           float saturation,
           float vibrance,
           float cast,
           float gain_r,
           float gain_g,
           float gain_b) {
            
            auto buf = buffer.mutable_unchecked<1>();
            uint8_t* data = buf.mutable_data(0);
            
            iphoto::kernels::apply_color_adjustments_inplace(
                data, width, height, bytes_per_line,
                saturation, vibrance, cast,
                gain_r, gain_g, gain_b
            );
        },
        py::arg("buffer"),
        py::arg("width"),
        py::arg("height"),
        py::arg("bytes_per_line"),
        py::arg("saturation"),
        py::arg("vibrance"),
        py::arg("cast"),
        py::arg("gain_r"),
        py::arg("gain_g"),
        py::arg("gain_b"),
        "Apply color adjustments in-place"
    );
    
    // Expose utility functions for testing
    m.def("clamp", &iphoto::math::clamp, "Clamp value to range");
    m.def("clamp01", &iphoto::math::clamp01, "Clamp value to [0, 1]");
    m.def("float_to_uint8", &iphoto::math::float_to_uint8, "Convert float to uint8");
}
```

### 2.4 CMake 构建配置

#### `cpp/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.15)
project(iphoto_cpp_filters VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Source files
set(SOURCES
    src/algorithms.cpp
    src/image_kernels.cpp
    src/bw_algorithms.cpp
    src/math_utils.cpp
    src/bindings.cpp
)

# Create Python module
pybind11_add_module(_cpp_filters ${SOURCES})

target_include_directories(_cpp_filters
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_cpp_filters PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -ffast-math
        -funroll-loops
        -fvectorize
    )
elseif(MSVC)
    target_compile_options(_cpp_filters PRIVATE
        /W4
        /fp:fast
        /arch:AVX2  # Use AVX2 if available
    )
endif()

# Install target
install(TARGETS _cpp_filters
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)

# Optional: Build tests
option(BUILD_TESTS "Build C++ tests" OFF)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

## 3. Python 包装器实现

### 3.1 创建 `cpp_executor.py`

```python
"""C++ accelerated executor for image adjustments.

This module provides a Python interface to the C++ implementation
of image processing algorithms via pybind11.
"""

from __future__ import annotations

import logging
import numpy as np
from PySide6.QtGui import QImage

from .utils import _resolve_pixel_buffer

logger = logging.getLogger(__name__)

# Try to import the compiled C++ module
_CPP_AVAILABLE = False
_cpp_filters = None

try:
    from . import _cpp_filters
    _CPP_AVAILABLE = True
    logger.info("Loaded C++ compiled image filters.")
except ImportError as e:
    logger.debug(f"C++ compiled module not found: {e}")
    _cpp_filters = None


def is_cpp_available() -> bool:
    """Check if C++ filters are available."""
    return _CPP_AVAILABLE


def apply_adjustments_fast_qimage(
    image: QImage,
    width: int,
    height: int,
    bytes_per_line: int,
    exposure_term: float,
    brightness_term: float,
    brilliance_strength: float,
    highlights: float,
    shadows: float,
    contrast_factor: float,
    black_point: float,
    saturation: float,
    vibrance: float,
    cast: float,
    gain_r: float,
    gain_g: float,
    gain_b: float,
    apply_bw: bool,
    bw_intensity: float,
    bw_neutrals: float,
    bw_tone: float,
    bw_grain: float,
) -> None:
    """Apply adjustments using C++ implementation."""
    
    if not _CPP_AVAILABLE or _cpp_filters is None:
        raise RuntimeError(
            "C++ filters module is not available. "
            "Please ensure the module is compiled correctly."
        )
    
    view, buffer_guard = _resolve_pixel_buffer(image)
    _ = buffer_guard
    
    if getattr(view, "readonly", False):
        raise BufferError("QImage pixel buffer is read-only")
    
    if width <= 0 or height <= 0:
        return
    
    expected_size = bytes_per_line * height
    buffer = np.frombuffer(view, dtype=np.uint8, count=expected_size)
    if buffer.size < expected_size:
        raise BufferError("QImage pixel buffer is smaller than expected")
    
    apply_color = abs(saturation) > 1e-6 or abs(vibrance) > 1e-6 or cast > 1e-6
    
    _cpp_filters.apply_adjustments_fast(
        buffer,
        width,
        height,
        bytes_per_line,
        exposure_term,
        brightness_term,
        brilliance_strength,
        highlights,
        shadows,
        contrast_factor,
        black_point,
        saturation,
        vibrance,
        cast,
        gain_r,
        gain_g,
        gain_b,
        apply_color,
        apply_bw,
        bw_intensity,
        bw_neutrals,
        bw_tone,
        bw_grain,
    )


def apply_color_adjustments_inplace_qimage(
    image: QImage,
    saturation: float,
    vibrance: float,
    cast: float,
    gain_r: float,
    gain_g: float,
    gain_b: float,
) -> None:
    """Apply only color adjustments using C++ implementation."""
    
    if not _CPP_AVAILABLE or _cpp_filters is None:
        raise RuntimeError(
            "C++ filters module is not available. "
            "Please ensure the module is compiled correctly."
        )
    
    if image.isNull():
        return
    
    apply_color = abs(saturation) > 1e-6 or abs(vibrance) > 1e-6 or cast > 1e-6
    if not apply_color:
        return
    
    view, guard = _resolve_pixel_buffer(image)
    _ = guard
    
    if getattr(view, "readonly", False):
        raise BufferError("QImage pixel buffer is read-only")
    
    width = image.width()
    height = image.height()
    bytes_per_line = image.bytesPerLine()
    
    if width <= 0 or height <= 0:
        return
    
    expected_size = bytes_per_line * height
    buffer = np.frombuffer(view, dtype=np.uint8, count=expected_size)
    if buffer.size < expected_size:
        raise BufferError("QImage pixel buffer is smaller than expected")
    
    _cpp_filters.apply_color_adjustments_inplace(
        buffer,
        width,
        height,
        bytes_per_line,
        saturation,
        vibrance,
        cast,
        gain_r,
        gain_g,
        gain_b,
    )
```

## 4. 需要修改的现有文件

### 4.1 修改 `src/iPhoto/core/filters/jit_executor.py`

在策略解析部分添加 C++ 执行器优先级：

```python
# Execution Strategy Resolution
# -----------------------------
# 1. C++ (pybind11): Preferred for production (fastest, no JIT overhead)
# 2. AOT (Ahead-Of-Time): Fallback if C++ not available
# 3. JIT (Just-In-Time): Development fallback
# 4. NumPy: Last resort

_CPP_AVAILABLE = False
_AOT_AVAILABLE = False
_JIT_AVAILABLE = False

# 1. Try C++ first
try:
    from .cpp_executor import (
        is_cpp_available,
        apply_adjustments_fast_qimage as _cpp_apply_adjustments,
        apply_color_adjustments_inplace_qimage as _cpp_apply_color,
    )
    
    if is_cpp_available():
        _apply_adjustments_fast = _cpp_apply_adjustments
        _apply_color_adjustments_inplace = _cpp_apply_color
        _CPP_AVAILABLE = True
        logger.info("Using C++ compiled filters (highest performance)")
except ImportError:
    logger.debug("C++ module not available, trying AOT...")

# 2. Try AOT if C++ not available
if not _CPP_AVAILABLE:
    # ... (existing AOT code)
    
# 3. Try JIT if neither C++ nor AOT available
# ... (existing JIT code)
```

### 4.2 修改 `src/iPhoto/core/color_resolver.py`

对于 `color_resolver.py` 中的 `_clamp` 函数，可以选择：

**选项 A**: 直接使用 Python 实现（性能影响很小）

```python
def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp value to range [minimum, maximum]."""
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
```

**选项 B**: 从 C++ 模块导入（如果可用）

```python
try:
    from .filters._cpp_filters import clamp as _clamp
except ImportError:
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        if value < minimum:
            return minimum
        if value > maximum:
            return maximum
        return value
```

### 4.3 修改 `pyproject.toml`

```toml
[build-system]
requires = [
    "setuptools>=68",
    "wheel",
    "pybind11>=2.11.0",
    "cmake>=3.15",
]
build-backend = "setuptools.build_meta"

[project]
# ... existing fields ...
dependencies = [
    # ... other dependencies ...
    "numpy>=2.3.4",
    # Remove: "numba>=0.60",  # No longer needed!
    # ... rest of dependencies ...
]

[project.optional-dependencies]
dev = [
    # ... existing dev deps ...
    "pybind11>=2.11.0",
    "cmake>=3.15",
]

# Add build extension configuration
[tool.setuptools.dynamic]
version = {attr = "iPhoto.__version__"}

[tool.setuptools.package-data]
iPhoto = ["core/filters/*.so", "core/filters/*.pyd"]
```

### 4.4 创建 `setup.py` 用于构建 C++ 扩展

```python
"""Setup script for building C++ extensions with pybind11."""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Extension that uses CMake to build."""
    
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build extension using CMake."""
    
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return
        
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # CMake configuration arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]
        
        build_args = ["--config", "Release"]
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )
        
        # Build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=self.build_temp
        )


setup(
    ext_modules=[
        CMakeExtension("iPhoto.core.filters._cpp_filters", sourcedir="cpp")
    ],
    cmdclass={"build_ext": CMakeBuild},
)
```

## 5. 迁移步骤

### 5.1 阶段 1: 准备工作

1. **安装开发依赖**
   ```bash
   pip install pybind11 cmake
   # 或者在 Linux 上
   sudo apt-get install cmake build-essential
   ```

2. **创建 C++ 项目结构**
   ```bash
   mkdir -p cpp/{include/iphoto,src,tests}
   ```

3. **创建所有头文件**
   - 按照 2.2 节创建所有 `.hpp` 文件

### 5.2 阶段 2: 实现 C++ 代码

1. **实现核心算法**
   - 创建 `cpp/src/math_utils.cpp`
   - 创建 `cpp/src/algorithms.cpp`
   - 逐一移植 `algorithms.py` 中的函数

2. **实现图像处理内核**
   - 创建 `cpp/src/image_kernels.cpp`
   - 移植 `jit_kernels.py` 中的两个主函数

3. **创建 pybind11 绑定**
   - 创建 `cpp/src/bindings.cpp`
   - 确保参数类型和函数签名匹配

### 5.3 阶段 3: 构建和测试

1. **配置 CMake**
   ```bash
   cd cpp
   mkdir build
   cd build
   cmake ..
   make
   ```

2. **复制编译产物**
   ```bash
   # Linux
   cp _cpp_filters.cpython-*.so ../../src/iPhoto/core/filters/
   
   # Windows
   copy _cpp_filters.*.pyd ..\..\src\iPhoto\core\filters\
   ```

3. **测试 Python 绑定**
   ```python
   from iPhoto.core.filters import _cpp_filters
   import numpy as np
   
   # 测试基本函数
   assert _cpp_filters.clamp(1.5, 0.0, 1.0) == 1.0
   assert _cpp_filters.clamp01(1.5) == 1.0
   
   # 测试图像处理
   buffer = np.zeros(1024, dtype=np.uint8)
   _cpp_filters.apply_color_adjustments_inplace(
       buffer, 10, 10, 40,  # 10x10 image, 40 bytes per line (RGBA)
       0.5, 0.3, 0.1,       # saturation, vibrance, cast
       1.0, 1.0, 1.0        # gains
   )
   ```

### 5.4 阶段 4: 集成到项目

1. **创建 `cpp_executor.py`**
   - 按照 3.1 节实现

2. **修改 `jit_executor.py`**
   - 按照 4.1 节添加 C++ 优先级

3. **更新构建配置**
   - 修改 `pyproject.toml`
   - 创建 `setup.py`

4. **测试集成**
   ```bash
   # 运行现有测试确保兼容性
   pytest tests/ui/widgets/test_filmstrip_performance.py
   pytest tests/
   ```

### 5.5 阶段 5: 性能验证

1. **基准测试**
   ```python
   import time
   import numpy as np
   from PySide6.QtGui import QImage
   from iPhoto.core.filters.cpp_executor import apply_adjustments_fast_qimage
   
   # 创建测试图像
   img = QImage(4000, 3000, QImage.Format.Format_RGBA8888)
   img.fill(0)
   
   # 性能测试
   start = time.perf_counter()
   for _ in range(10):
       apply_adjustments_fast_qimage(
           img, 4000, 3000, img.bytesPerLine(),
           0.1, 0.2, 0.1, 0.1, 0.1, 1.2, 0.0,
           0.5, 0.3, 0.1, 1.0, 1.0, 1.0,
           False, 0.5, 0.5, 0.5, 0.0
       )
   elapsed = time.perf_counter() - start
   print(f"Average time: {elapsed / 10 * 1000:.2f} ms")
   ```

2. **对比测试**
   - 与 numba JIT 版本对比
   - 与 AOT 版本对比
   - 确保 C++ 版本至少同样快或更快

### 5.6 阶段 6: 清理和文档

1. **移除 numba 依赖**
   - 从 `pyproject.toml` 移除 numba
   - 标记旧文件为已弃用

2. **更新文档**
   - 更新 `README.md`
   - 更新构建说明
   - 添加 C++ 开发文档

3. **添加废弃警告**
   ```python
   # 在 jit_kernels.py 顶部
   import warnings
   warnings.warn(
       "jit_kernels.py is deprecated. "
       "The project now uses C++ implementation via pybind11.",
       DeprecationWarning,
       stacklevel=2
   )
   ```

## 6. 优化建议

### 6.1 SIMD 优化

使用 SIMD 指令加速像素处理：

```cpp
#ifdef __AVX2__
#include <immintrin.h>

// 在 image_kernels.cpp 中
// Process 8 pixels at once using AVX2
void apply_adjustments_fast_simd(/* ... */) {
    // AVX2 implementation for parallel pixel processing
    // ...
}
#endif
```

### 6.2 多线程优化

使用 OpenMP 并行化行处理：

```cpp
#include <omp.h>

void apply_adjustments_fast(/* ... */) {
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        // Process row
    }
}
```

在 CMakeLists.txt 中启用：

```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_cpp_filters PRIVATE OpenMP::OpenMP_CXX)
endif()
```

### 6.3 编译器优化标志

```cmake
# 针对不同平台的优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(_cpp_filters PRIVATE
            -march=native      # 使用本机 CPU 指令集
            -mtune=native      # 针对本机 CPU 优化
            -ffast-math        # 快速数学运算
            -funroll-loops     # 循环展开
        )
    endif()
endif()
```

## 7. 跨平台支持

### 7.1 Windows 构建

使用 Visual Studio:

```batch
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### 7.2 macOS 构建

```bash
brew install cmake pybind11
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### 7.3 Linux 构建

```bash
sudo apt-get install cmake build-essential python3-dev
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 8. 持续集成

### 8.1 GitHub Actions 工作流

创建 `.github/workflows/build-cpp.yml`:

```yaml
name: Build C++ Extensions

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.12']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pybind11 cmake pytest numpy PySide6
    
    - name: Build C++ extension
      run: |
        cd cpp
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        cmake --build . --config Release
    
    - name: Run tests
      run: |
        pytest tests/ -v
```

## 9. 性能对比

预期性能提升（相比 numba JIT）：

| 操作 | Numba JIT | C++ (无优化) | C++ (O3) | C++ (O3 + SIMD) | C++ (O3 + SIMD + OpenMP) |
|------|-----------|-------------|----------|----------------|------------------------|
| 4K 图像全调整 | 100% | 120% | 180% | 350% | 700% (4核) |
| 颜色调整 | 100% | 140% | 200% | 400% | 800% (4核) |
| 编译时间 | 2-5秒 | 0秒 | 0秒 | 0秒 | 0秒 |
| 首次运行 | 慢（JIT） | 快 | 快 | 快 | 快 |

**优势**:
- 无 JIT 编译延迟
- 更好的编译时优化
- 可使用 SIMD 指令
- 可多线程并行
- 更小的内存占用

## 10. 故障排除

### 10.1 常见问题

**问题 1**: CMake 找不到 pybind11

```bash
pip install "pybind11[global]"
# 或
cmake .. -Dpybind11_DIR=/path/to/pybind11/share/cmake/pybind11
```

**问题 2**: Python 找不到编译的模块

```python
# 确保模块在正确位置
import sys
print(sys.path)
# 应包含 src/iPhoto/core/filters/
```

**问题 3**: 符号未定义错误

- 检查所有 `.cpp` 文件是否都在 CMakeLists.txt 的 SOURCES 中
- 确保头文件正确包含

### 10.2 调试技巧

启用调试符号：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```

使用 GDB/LLDB 调试：

```bash
gdb --args python -c "from iPhoto.core.filters import _cpp_filters; ..."
```

## 11. 总结

### 11.1 迁移检查清单

- [ ] 创建 C++ 项目结构
- [ ] 实现所有算法函数
- [ ] 创建 pybind11 绑定
- [ ] 配置 CMake 构建系统
- [ ] 实现 Python 包装器
- [ ] 更新执行器优先级
- [ ] 运行测试套件
- [ ] 性能基准测试
- [ ] 更新文档
- [ ] 移除 numba 依赖
- [ ] 配置 CI/CD

### 11.2 优势总结

1. **性能**: 编译时优化，SIMD，多线程
2. **兼容性**: 不依赖 LLVM，更好的跨平台支持
3. **可维护性**: 标准 C++，更多工具支持
4. **未来扩展**: 可接入 CUDA/Metal 等 GPU 加速

### 11.3 后续工作

- 添加 GPU 加速支持（CUDA/OpenCL）
- 实现更多 SIMD 优化路径
- 添加 ARM NEON 支持
- 创建性能分析工具
- 添加更多单元测试

## 附录 A: 完整文件清单

### A.1 新增文件

1. `cpp/include/iphoto/types.hpp`
2. `cpp/include/iphoto/math_utils.hpp`
3. `cpp/include/iphoto/algorithms.hpp`
4. `cpp/include/iphoto/image_kernels.hpp`
5. `cpp/src/math_utils.cpp`
6. `cpp/src/algorithms.cpp`
7. `cpp/src/bw_algorithms.cpp`
8. `cpp/src/image_kernels.cpp`
9. `cpp/src/bindings.cpp`
10. `cpp/CMakeLists.txt`
11. `cpp/tests/test_algorithms.cpp`
12. `cpp/tests/CMakeLists.txt`
13. `cpp/README.md`
14. `src/iPhoto/core/filters/cpp_executor.py`
15. `setup.py`
16. `.github/workflows/build-cpp.yml`
17. `docs/pybind11_migration_guide.md` (本文档)

### A.2 需要修改的文件

1. `src/iPhoto/core/filters/jit_executor.py` - 添加 C++ 执行器支持
2. `src/iPhoto/core/color_resolver.py` - 移除或替换 numba 装饰器
3. `pyproject.toml` - 更新依赖和构建配置
4. `README.md` - 更新构建说明
5. `CONTRIBUTING.md` - 添加 C++ 开发指南

### A.3 可选废弃的文件

这些文件在迁移后可以标记为废弃，但应保留作为参考：

1. `src/iPhoto/core/filters/build_jit.py`
2. `src/iPhoto/core/filters/jit_kernels.py`
3. `src/iPhoto/core/filters/algorithms.py` (numba 版本)

## 附录 B: 参考资源

### B.1 官方文档

- [pybind11 文档](https://pybind11.readthedocs.io/)
- [CMake 文档](https://cmake.org/documentation/)
- [NumPy C API](https://numpy.org/doc/stable/reference/c-api/)

### B.2 性能优化

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
- [OpenMP 规范](https://www.openmp.org/specifications/)

### B.3 相关项目

- [OpenCV](https://github.com/opencv/opencv) - 图像处理参考
- [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) - SIMD 优化示例
- [scikit-image](https://github.com/scikit-image/scikit-image) - Python/C 混合实现参考

---

**文档版本**: 1.0.0  
**最后更新**: 2026-02-13  
**维护者**: iPhoto 开发团队
