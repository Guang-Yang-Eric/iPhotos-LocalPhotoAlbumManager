# Pybind11 Migration Guide: Replacing Numba JIT with C++ Extensions

> **Version**: 1.0
> **Date**: 2026-02-11
> **Target Project**: iPhoto - LocalPhotoAlbumManager
> **Current Acceleration**: Numba JIT / AOT
> **Target Acceleration**: pybind11 + C++

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Current Numba Architecture](#2-current-numba-architecture)
3. [Target Architecture with pybind11](#3-target-architecture-with-pybind11)
4. [New File Structure](#4-new-file-structure)
5. [C++ Source Files Specification](#5-c-source-files-specification)
6. [Python Files to Modify](#6-python-files-to-modify)
7. [Build System Integration](#7-build-system-integration)
8. [Function Migration Reference](#8-function-migration-reference)
9. [Migration Steps](#9-migration-steps)
10. [Testing Strategy](#10-testing-strategy)
11. [Fallback and Compatibility](#11-fallback-and-compatibility)
12. [Dependency Changes](#12-dependency-changes)
13. [FAQ](#13-faq)

---

## 1. Overview and Motivation

### 1.1 Why Replace Numba with pybind11 + C++?

The current project uses Numba for JIT/AOT compilation of performance-critical image processing
algorithms. While Numba provides excellent development-time convenience, it introduces several
challenges:

| Issue | Description |
|-------|-------------|
| **Large dependency footprint** | numba + llvmlite add ~300 MB to the distribution |
| **Python version coupling** | Numba often lags behind new Python releases |
| **Nuitka/PyInstaller friction** | JIT compilation is incompatible with compiled/frozen builds |
| **Limited C++ ecosystem access** | Cannot leverage SIMD intrinsics, OpenMP, or existing C++ image processing libraries |
| **Complex AOT build step** | The current build_jit.py AOT step adds build complexity |

### 1.2 Benefits of pybind11

- **Minimal overhead**: Near-zero call overhead for C++ functions
- **No runtime compiler needed**: Compiled once, runs everywhere
- **Full C++ power**: SIMD, OpenMP, template metaprogramming
- **Smaller distribution**: Only the compiled .so/.pyd extension ships
- **Broad compatibility**: Works with Nuitka, PyInstaller, and all major Python versions
- **NumPy interop**: Native support for NumPy arrays via pybind11::array_t

---

## 2. Current Numba Architecture

### 2.1 Execution Strategy (Three-Tier Fallback)

    jit_executor.py
      |
      +-- (1) AOT: _jit_compiled (pre-compiled C extension via numba.pycc)
      |
      +-- (2) JIT: jit_kernels.py (runtime Numba compilation)
      |
      +-- (3) NumPy: numpy_executor.py (pure NumPy vectorized fallback)

### 2.2 Files Using Numba

| File | Numba Usage | Functions |
|------|------------|-----------|
| src/iPhoto/core/filters/algorithms.py | @jit(nopython=True, inline="always") | 10 functions (core math) |
| src/iPhoto/core/filters/jit_kernels.py | @jit(nopython=True, cache=True) | 2 kernel functions |
| src/iPhoto/core/filters/jit_executor.py | Strategy selector | Imports AOT/JIT/NumPy |
| src/iPhoto/core/filters/build_jit.py | numba.pycc.CC AOT builder | Build script |
| src/iPhoto/core/color_resolver.py | @jit(nopython=True, inline="always") | 1 function (_clamp) |
| demo/curve/curve.py | Optional @jit import | Demo utility |

### 2.3 Complete Function Inventory

#### algorithms.py - 10 Numba-decorated functions

| Function | Signature | Description |
|----------|-----------|-------------|
| _clamp01(value) | (f64) -> f64 | Clamp to [0.0, 1.0] |
| _clamp(value, min_val, max_val) | (f64, f64, f64) -> f64 | Clamp to [min, max] |
| _mix(a, b, t) | (f64, f64, f64) -> f64 | Linear interpolation |
| _float_to_uint8(value) | (f64) -> i32 | Float [0,1] to uint8 |
| _grain_noise(x, y, width, height) | (i32, i32, i32, i32) -> f64 | Deterministic noise |
| _contrast_tone_curve(value, tone) | (f64, f64) -> f64 | Sigmoid tone curve |
| _gamma_neutral(value, neutrals) | (f64, f64) -> f64 | Gamma adjustment |
| _apply_channel_adjustments(...) | (f64 x8) -> f64 | Tone curve per channel |
| _apply_color_transform(...) | (f64 x9) -> (f64, f64, f64) | RGB color adjustments |
| _apply_bw_channels(...) | (f64 x8) -> (f64, f64, f64) | B&W effect |

#### jit_kernels.py - 2 Numba-decorated kernel functions

| Function | Signature | Description |
|----------|-----------|-------------|
| _apply_adjustments_fast(...) | (u8[:], i64, i64, i64, f64x13, bool, bool, f64x4) -> void | Full adjustment kernel |
| _apply_color_adjustments_inplace(...) | (u8[:], i64, i64, i64, f64x6) -> void | Color-only kernel |

#### color_resolver.py - 1 Numba-decorated function

| Function | Signature | Description |
|----------|-----------|-------------|
| _clamp(value, minimum, maximum) | (f64, f64, f64) -> f64 | Value clamping |

---

## 3. Target Architecture with pybind11

### 3.1 New Execution Strategy

    jit_executor.py (modified)
      |
      +-- (1) C++ Extension: _cpp_filters (pybind11 compiled module)
      |
      +-- (2) NumPy: numpy_executor.py (unchanged fallback)

The three-tier strategy simplifies to two tiers:
- **Primary**: C++ compiled extension via pybind11 (_cpp_filters)
- **Fallback**: NumPy vectorized implementation (existing numpy_executor.py)

### 3.2 Architecture Diagram

    Python Layer                          C++ Layer
    ==============================        ==============================
    jit_executor.py                       cpp/
      import _cpp_filters  -------->        src/
                                              algorithms.h / algorithms.cpp
      _apply_adjustments_fast()               kernels.h    / kernels.cpp
      _apply_color_adjustments_inplace()      bindings.cpp (pybind11 module)
                                            CMakeLists.txt

---

## 4. New File Structure

### 4.1 Files to Add (C++ Extension)

    src/iPhoto/core/filters/cpp/
     +-- CMakeLists.txt               # CMake build configuration
     +-- src/
     |    +-- algorithms.h            # Header: core math functions
     |    +-- algorithms.cpp          # Implementation: core math functions
     |    +-- kernels.h               # Header: pixel processing kernels
     |    +-- kernels.cpp             # Implementation: pixel processing kernels
     |    +-- bindings.cpp            # pybind11 module definition
     +-- tests/
          +-- test_algorithms.cpp     # C++ unit tests (optional, Google Test)

### 4.2 Files to Modify (Python Side)

| File | Change Type | Description |
|------|-------------|-------------|
| src/iPhoto/core/filters/jit_executor.py | **Major rewrite** | Replace AOT/JIT tiers with pybind11 C++ import |
| src/iPhoto/core/filters/algorithms.py | **Remove numba** | Remove @jit decorators; keep as pure Python reference |
| src/iPhoto/core/filters/jit_kernels.py | **Delete** | No longer needed (C++ replaces JIT kernels) |
| src/iPhoto/core/filters/build_jit.py | **Delete** | No longer needed (CMake replaces AOT build) |
| src/iPhoto/core/color_resolver.py | **Minor edit** | Remove numba import; use plain Python _clamp |
| demo/curve/curve.py | **Minor edit** | Remove numba import; use plain decorator fallback |
| pyproject.toml | **Edit dependencies** | Remove numba>=0.60; add pybind11>=2.12 as build dep |
| docs/BUILD_EXE.md | **Update** | Replace AOT build instructions with C++ build instructions |

### 4.3 Files to Delete

| File | Reason |
|------|--------|
| src/iPhoto/core/filters/jit_kernels.py | Replaced by cpp/src/kernels.cpp |
| src/iPhoto/core/filters/build_jit.py | Replaced by cpp/CMakeLists.txt |
| src/iPhoto/core/filters/_jit_compiled*.so / .pyd | Old AOT artifacts (if any exist) |

### 4.4 Complete Directory Overview After Migration

    src/iPhoto/core/filters/
     +-- __init__.py                  # (unchanged)
     +-- algorithms.py                # (modified: remove @jit decorators)
     +-- facade.py                    # (unchanged)
     +-- fallback_executor.py         # (unchanged)
     +-- jit_executor.py              # (modified: import _cpp_filters instead of numba)
     +-- numpy_executor.py            # (unchanged: remains as fallback)
     +-- pillow_executor.py           # (unchanged)
     +-- utils.py                     # (unchanged)
     +-- cpp/                         # (NEW: C++ extension source)
     |    +-- CMakeLists.txt
     |    +-- src/
     |    |    +-- algorithms.h
     |    |    +-- algorithms.cpp
     |    |    +-- kernels.h
     |    |    +-- kernels.cpp
     |    |    +-- bindings.cpp
     |    +-- tests/
     |         +-- test_algorithms.cpp

---

## 5. C++ Source Files Specification

### 5.1 algorithms.h - Core Math Functions Header

This header declares all pure math functions that were previously in algorithms.py.

```cpp
// src/iPhoto/core/filters/cpp/src/algorithms.h
#pragma once

#include <cmath>
#include <tuple>
#include <cstdint>

namespace iphoto {

// Utility functions
inline double clamp01(double value);
inline double clamp(double value, double min_val, double max_val);
inline double mix(double a, double b, double t);
inline uint8_t float_to_uint8(double value);

// Noise and tone
double grain_noise(int x, int y, int width, int height);
double contrast_tone_curve(double value, double tone);
double gamma_neutral(double value, double neutrals);

// Channel adjustments
double apply_channel_adjustments(
    double value, double exposure, double brightness, double brilliance,
    double highlights, double shadows, double contrast_factor, double black_point
);

// Color transform (returns r, g, b)
std::tuple<double, double, double> apply_color_transform(
    double r, double g, double b,
    double saturation, double vibrance, double cast,
    double gain_r, double gain_g, double gain_b
);

// Black and White channels (returns r, g, b)
std::tuple<double, double, double> apply_bw_channels(
    double r, double g, double b,
    double intensity, double neutrals, double tone,
    double grain, double noise
);

} // namespace iphoto
```

### 5.2 algorithms.cpp - Core Math Functions Implementation

```cpp
// src/iPhoto/core/filters/cpp/src/algorithms.cpp
#include "algorithms.h"
#include <algorithm>

namespace iphoto {

inline double clamp01(double value) {
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

inline double clamp(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

inline double mix(double a, double b, double t) {
    t = clamp01(t);
    return a * (1.0 - t) + b * t;
}

inline uint8_t float_to_uint8(double value) {
    double scaled = std::round(value * 255.0);
    if (scaled < 0.0) return 0;
    if (scaled > 255.0) return 255;
    return static_cast<uint8_t>(scaled);
}

double grain_noise(int x, int y, int width, int height) {
    if (width <= 0 || height <= 0) return 0.5;
    double u = static_cast<double>(x) / static_cast<double>(std::max(width - 1, 1));
    double v = static_cast<double>(y) / static_cast<double>(std::max(height - 1, 1));
    double seed = u * 12.9898 + v * 78.233;
    double noise = std::sin(seed) * 43758.5453;
    double fraction = noise - std::floor(noise);
    return clamp01(fraction);
}

double contrast_tone_curve(double value, double tone) {
    tone = clamp01(tone);
    double t = tone - 0.5;
    double factor = (t >= 0.0)
        ? mix(1.0, 2.2, t * 2.0)
        : mix(1.0, 0.6, -t * 2.0);
    double x = clamp(value, 0.0, 1.0);
    constexpr double eps = 1e-6;
    double pos = clamp(x, eps, 1.0 - eps);
    double logit = std::log(pos / std::max(eps, 1.0 - pos));
    double y = 1.0 / (1.0 + std::exp(-logit * factor));
    return clamp(y, 0.0, 1.0);
}

double gamma_neutral(double value, double neutrals) {
    neutrals = clamp01(neutrals);
    double n = 0.6 * (neutrals - 0.5);
    double gamma_val = std::pow(2.0, -n * 2.0);
    return clamp(std::pow(clamp(value, 0.0, 1.0), gamma_val), 0.0, 1.0);
}

double apply_channel_adjustments(
    double value, double exposure, double brightness, double brilliance,
    double highlights, double shadows, double contrast_factor, double black_point
) {
    double adjusted = value + exposure + brightness;

    double mid_distance = value - 0.5;
    adjusted += brilliance * (1.0 - (mid_distance * 2.0) * (mid_distance * 2.0));

    if (adjusted > 0.65) {
        double ratio = (adjusted - 0.65) / 0.35;
        adjusted += highlights * ratio;
    } else if (adjusted < 0.35) {
        double ratio = (0.35 - adjusted) / 0.35;
        adjusted += shadows * ratio;
    }

    adjusted = (adjusted - 0.5) * contrast_factor + 0.5;

    if (black_point > 0.0) {
        adjusted -= black_point * (1.0 - adjusted);
    } else if (black_point < 0.0) {
        adjusted -= black_point * adjusted;
    }

    return clamp01(adjusted);
}

std::tuple<double, double, double> apply_color_transform(
    double r, double g, double b,
    double saturation, double vibrance, double cast,
    double gain_r, double gain_g, double gain_b
) {
    double mix_r = (1.0 - cast) + gain_r * cast;
    double mix_g = (1.0 - cast) + gain_g * cast;
    double mix_b = (1.0 - cast) + gain_b * cast;
    r *= mix_r; g *= mix_g; b *= mix_b;

    double luma = 0.299 * r + 0.587 * g + 0.114 * b;
    double chroma_r = r - luma;
    double chroma_g = g - luma;
    double chroma_b = b - luma;

    double sat_amt = 1.0 + saturation;
    double vib_amt = 1.0 + vibrance;
    double w = 1.0 - clamp(std::abs(luma - 0.5) * 2.0, 0.0, 1.0);
    double chroma_scale = sat_amt * (1.0 + (vib_amt - 1.0) * w);
    chroma_r *= chroma_scale;
    chroma_g *= chroma_scale;
    chroma_b *= chroma_scale;

    r = clamp(luma + chroma_r, 0.0, 1.0);
    g = clamp(luma + chroma_g, 0.0, 1.0);
    b = clamp(luma + chroma_b, 0.0, 1.0);
    return {r, g, b};
}

std::tuple<double, double, double> apply_bw_channels(
    double r, double g, double b,
    double intensity, double neutrals, double tone,
    double grain, double noise
) {
    intensity = clamp01(intensity);
    neutrals = clamp01(neutrals);
    tone = clamp01(tone);
    grain = clamp01(grain);
    noise = clamp01(noise);

    double luma = clamp(0.2126 * r + 0.7152 * g + 0.0722 * b, 0.0, 1.0);

    double soft_base = clamp(std::pow(luma, 0.82), 0.0, 1.0);
    double soft_curve = contrast_tone_curve(soft_base, 0.0);
    double g_soft = (soft_curve + soft_base) * 0.5;
    double g_neutral = luma;
    double g_rich = contrast_tone_curve(
        clamp(std::pow(luma, 1.0 / 1.22), 0.0, 1.0), 0.35);

    double gray;
    if (intensity >= 0.5) {
        double blend = (intensity - 0.5) / 0.5;
        gray = mix(g_neutral, g_rich, blend);
    } else {
        double blend = (0.5 - intensity) / 0.5;
        gray = mix(g_soft, g_neutral, blend);
    }

    gray = gamma_neutral(gray, neutrals);
    gray = contrast_tone_curve(gray, tone);

    if (grain > 1e-6) {
        gray += (noise - 0.5) * 0.2 * grain;
    }

    double clamped = clamp01(gray);
    return {clamped, clamped, clamped};
}

} // namespace iphoto
```

### 5.3 kernels.h - Pixel Processing Kernels Header

```cpp
// src/iPhoto/core/filters/cpp/src/kernels.h
#pragma once

#include <cstdint>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace iphoto {

void apply_adjustments_fast(
    py::array_t<uint8_t> buffer,
    int64_t width, int64_t height, int64_t bytes_per_line,
    double exposure_term, double brightness_term, double brilliance_strength,
    double highlights, double shadows, double contrast_factor, double black_point,
    double saturation, double vibrance, double cast,
    double gain_r, double gain_g, double gain_b,
    bool apply_color, bool apply_bw,
    double bw_intensity, double bw_neutrals, double bw_tone, double bw_grain
);

void apply_color_adjustments_inplace(
    py::array_t<uint8_t> buffer,
    int64_t width, int64_t height, int64_t bytes_per_line,
    double saturation, double vibrance, double cast,
    double gain_r, double gain_g, double gain_b
);

} // namespace iphoto
```

### 5.4 kernels.cpp - Pixel Processing Kernels Implementation

```cpp
// src/iPhoto/core/filters/cpp/src/kernels.cpp
#include "kernels.h"
#include "algorithms.h"
#include <cmath>

namespace iphoto {

void apply_adjustments_fast(
    py::array_t<uint8_t> buffer,
    int64_t width, int64_t height, int64_t bytes_per_line,
    double exposure_term, double brightness_term, double brilliance_strength,
    double highlights, double shadows, double contrast_factor, double black_point,
    double saturation, double vibrance, double cast,
    double gain_r, double gain_g, double gain_b,
    bool apply_color_flag, bool apply_bw,
    double bw_intensity, double bw_neutrals, double bw_tone, double bw_grain
) {
    if (width <= 0 || height <= 0) return;
    auto buf = buffer.mutable_unchecked<1>();

    for (int64_t y = 0; y < height; ++y) {
        int64_t row_offset = y * bytes_per_line;
        for (int64_t x = 0; x < width; ++x) {
            int64_t pixel_offset = row_offset + x * 4;
            double b = buf(pixel_offset)     / 255.0;
            double g = buf(pixel_offset + 1) / 255.0;
            double r = buf(pixel_offset + 2) / 255.0;

            r = apply_channel_adjustments(r, exposure_term, brightness_term,
                brilliance_strength, highlights, shadows, contrast_factor, black_point);
            g = apply_channel_adjustments(g, exposure_term, brightness_term,
                brilliance_strength, highlights, shadows, contrast_factor, black_point);
            b = apply_channel_adjustments(b, exposure_term, brightness_term,
                brilliance_strength, highlights, shadows, contrast_factor, black_point);

            if (apply_color_flag) {
                auto [cr, cg, cb] = apply_color_transform(
                    r, g, b, saturation, vibrance, cast, gain_r, gain_g, gain_b);
                r = cr; g = cg; b = cb;
            }
            if (apply_bw) {
                double noise_val = 0.0;
                if (std::abs(bw_grain) > 1e-6) {
                    noise_val = grain_noise(static_cast<int>(x), static_cast<int>(y),
                        static_cast<int>(width), static_cast<int>(height));
                }
                auto [bwr, bwg, bwb] = apply_bw_channels(
                    r, g, b, bw_intensity, bw_neutrals, bw_tone, bw_grain, noise_val);
                r = bwr; g = bwg; b = bwb;
            }
            buf(pixel_offset)     = float_to_uint8(b);
            buf(pixel_offset + 1) = float_to_uint8(g);
            buf(pixel_offset + 2) = float_to_uint8(r);
        }
    }
}

void apply_color_adjustments_inplace(
    py::array_t<uint8_t> buffer,
    int64_t width, int64_t height, int64_t bytes_per_line,
    double saturation, double vibrance, double cast,
    double gain_r, double gain_g, double gain_b
) {
    if (width <= 0 || height <= 0) return;
    bool need_color = std::abs(saturation) > 1e-6
                   || std::abs(vibrance) > 1e-6 || cast > 1e-6;
    if (!need_color) return;
    auto buf = buffer.mutable_unchecked<1>();

    for (int64_t y = 0; y < height; ++y) {
        int64_t row_offset = y * bytes_per_line;
        for (int64_t x = 0; x < width; ++x) {
            int64_t pixel_offset = row_offset + x * 4;
            double b = buf(pixel_offset)     / 255.0;
            double g = buf(pixel_offset + 1) / 255.0;
            double r = buf(pixel_offset + 2) / 255.0;
            auto [cr, cg, cb] = apply_color_transform(
                r, g, b, saturation, vibrance, cast, gain_r, gain_g, gain_b);
            buf(pixel_offset)     = float_to_uint8(cb);
            buf(pixel_offset + 1) = float_to_uint8(cg);
            buf(pixel_offset + 2) = float_to_uint8(cr);
        }
    }
}

} // namespace iphoto
```

### 5.5 bindings.cpp - pybind11 Module Definition

```cpp
// src/iPhoto/core/filters/cpp/src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kernels.h"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_filters, m) {
    m.doc() = "C++ accelerated image processing filters for iPhoto";

    m.def("_apply_adjustments_fast", &iphoto::apply_adjustments_fast,
        py::arg("buffer"), py::arg("width"), py::arg("height"),
        py::arg("bytes_per_line"),
        py::arg("exposure_term"), py::arg("brightness_term"),
        py::arg("brilliance_strength"),
        py::arg("highlights"), py::arg("shadows"),
        py::arg("contrast_factor"), py::arg("black_point"),
        py::arg("saturation"), py::arg("vibrance"), py::arg("cast"),
        py::arg("gain_r"), py::arg("gain_g"), py::arg("gain_b"),
        py::arg("apply_color"), py::arg("apply_bw"),
        py::arg("bw_intensity"), py::arg("bw_neutrals"),
        py::arg("bw_tone"), py::arg("bw_grain"),
        "Apply full image adjustments to a pixel buffer in-place."
    );

    m.def("_apply_color_adjustments_inplace",
        &iphoto::apply_color_adjustments_inplace,
        py::arg("buffer"), py::arg("width"), py::arg("height"),
        py::arg("bytes_per_line"),
        py::arg("saturation"), py::arg("vibrance"), py::arg("cast"),
        py::arg("gain_r"), py::arg("gain_g"), py::arg("gain_b"),
        "Apply color-only adjustments to a pixel buffer in-place."
    );
}
```

### 5.6 CMakeLists.txt - Build Configuration

```cmake
# src/iPhoto/core/filters/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(iphoto_cpp_filters LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(_cpp_filters
    src/algorithms.cpp
    src/kernels.cpp
    src/bindings.cpp
)
target_include_directories(_cpp_filters PRIVATE src)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_cpp_filters PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _cpp_filters DESTINATION .)
```

---

## 6. Python Files to Modify

### 6.1 jit_executor.py - Major Rewrite

Replace the three-tier (AOT/JIT/NumPy) fallback with a two-tier (C++/NumPy) strategy.

**Before** (current): Try AOT -> Try JIT -> Fall back to NumPy

**After** (new):

```python
"""C++ accelerated image adjustment executor."""
from __future__ import annotations
import logging
import numpy as np
from PySide6.QtGui import QImage
from .utils import _resolve_pixel_buffer

logger = logging.getLogger(__name__)
_CPP_AVAILABLE = False
_apply_adjustments_fast = None
_apply_color_adjustments_inplace = None

# 1. Try C++ extension (pybind11)
try:
    from . import _cpp_filters
    _apply_adjustments_fast = _cpp_filters._apply_adjustments_fast
    _apply_color_adjustments_inplace = _cpp_filters._apply_color_adjustments_inplace
    _CPP_AVAILABLE = True
    logger.info("Loaded C++ compiled image filters.")
except ImportError:
    logger.debug("C++ compiled module not found.")

# 2. Fall back to NumPy
if not _CPP_AVAILABLE:
    logger.warning("C++ extension unavailable. Falling back to NumPy.")
    from .numpy_executor import (
        apply_adjustments_buffer as _apply_adjustments_fast,
        apply_color_adjustments_inplace_buffer as _apply_color_adjustments_inplace,
    )
# Public API functions (apply_adjustments_fast_qimage, etc.) remain unchanged
```

### 6.2 algorithms.py - Remove Numba Decorators

- Remove the try/except numba import block (lines 9-24)
- Remove all @jit(...) decorators from every function (10 occurrences)
- Function bodies remain UNCHANGED

### 6.3 color_resolver.py - Remove Numba Import

- Remove `from numba import jit` (line 10)
- Remove @jit decorator from _clamp (line 313)

### 6.4 demo/curve/curve.py - Remove Numba Import

Replace the numba try/except block with a simple fallback decorator.

### 6.5 jit_kernels.py - DELETE

Entirely replaced by C++ kernels.

### 6.6 build_jit.py - DELETE

Replaced by CMake/pybind11 build system.

---

## 7. Build System Integration

### 7.1 setuptools with CMake Extension (Recommended)

Update pyproject.toml build-system requires:

```toml
[build-system]
requires = ["setuptools>=68", "wheel", "pybind11>=2.12", "cmake>=3.15"]
build-backend = "setuptools.build_meta"
```

Add a setup.py at the project root:

```python
import os, subprocess, sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=self.build_temp)

setup(
    ext_modules=[CMakeExtension("iPhoto.core.filters._cpp_filters",
                                sourcedir="src/iPhoto/core/filters/cpp")],
    cmdclass={"build_ext": CMakeBuild},
)
```

### 7.2 Manual Build (Development)

```bash
pip install pybind11
cd src/iPhoto/core/filters/cpp
mkdir -p build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3)
cmake --build . --config Release
cp _cpp_filters*.so ../..    # Linux/macOS
```

### 7.3 Build Verification

```bash
python -c "from iPhoto.core.filters._cpp_filters import _apply_adjustments_fast; print('OK')"
```

---

## 8. Function Migration Reference

| Python Source File | Python Function | C++ File | C++ Function |
|-------------------|----------------|----------|-------------|
| algorithms.py | _clamp01(value) | algorithms.cpp | iphoto::clamp01(value) |
| algorithms.py | _clamp(value, min_val, max_val) | algorithms.cpp | iphoto::clamp(value, min_val, max_val) |
| algorithms.py | _mix(a, b, t) | algorithms.cpp | iphoto::mix(a, b, t) |
| algorithms.py | _float_to_uint8(value) | algorithms.cpp | iphoto::float_to_uint8(value) |
| algorithms.py | _grain_noise(x, y, w, h) | algorithms.cpp | iphoto::grain_noise(x, y, w, h) |
| algorithms.py | _contrast_tone_curve(value, tone) | algorithms.cpp | iphoto::contrast_tone_curve(value, tone) |
| algorithms.py | _gamma_neutral(value, neutrals) | algorithms.cpp | iphoto::gamma_neutral(value, neutrals) |
| algorithms.py | _apply_channel_adjustments(...) | algorithms.cpp | iphoto::apply_channel_adjustments(...) |
| algorithms.py | _apply_color_transform(...) | algorithms.cpp | iphoto::apply_color_transform(...) |
| algorithms.py | _apply_bw_channels(...) | algorithms.cpp | iphoto::apply_bw_channels(...) |
| jit_kernels.py | _apply_adjustments_fast(...) | kernels.cpp | iphoto::apply_adjustments_fast(...) |
| jit_kernels.py | _apply_color_adjustments_inplace(...) | kernels.cpp | iphoto::apply_color_adjustments_inplace(...) |
| color_resolver.py | _clamp(value, min, max) | N/A (plain Python) | N/A |

> Note: color_resolver.py::_clamp is not in the hot path. Simply removing @jit is sufficient.

---

## 9. Migration Steps

### Phase 1: Prepare C++ Extension (No Python Changes)
1. Create directory structure: src/iPhoto/core/filters/cpp/src/
2. Write algorithms.h and algorithms.cpp: Port all 10 algorithm functions
3. Write kernels.h and kernels.cpp: Port the 2 kernel functions
4. Write bindings.cpp: Define the pybind11 module
5. Write CMakeLists.txt: Configure the build
6. Build and verify: Compile the extension and test import

### Phase 2: Numerical Validation
7. Write comparison tests for each C++ function vs Python/Numba
8. Run pixel-level regression tests on test images

### Phase 3: Integration
9. Modify jit_executor.py: Replace AOT/JIT with _cpp_filters import
10. Remove @jit decorators from algorithms.py
11. Remove numba import from color_resolver.py
12. Update demo/curve/curve.py: Remove numba import
13. Delete jit_kernels.py and build_jit.py

### Phase 4: Build System and Dependencies
14. Update pyproject.toml: Remove numba, add pybind11 to build requires
15. Create setup.py: Add CMake extension build integration
16. Update docs/BUILD_EXE.md: Replace AOT instructions

### Phase 5: Verification and Cleanup
17. Run full test suite
18. Test Nuitka/PyInstaller packaging
19. Performance benchmark
20. Remove any old AOT artifacts

---

## 10. Testing Strategy

### 10.1 Unit Tests for C++ Functions

```python
# tests/test_cpp_filters.py
import numpy as np
import pytest
from iPhoto.core.filters._cpp_filters import (
    _apply_adjustments_fast, _apply_color_adjustments_inplace)

class TestApplyAdjustmentsFast:
    def test_identity_transform(self):
        buffer = np.array([128, 64, 200, 255] * 4, dtype=np.uint8)
        original = buffer.copy()
        _apply_adjustments_fast(buffer, 2, 2, 8,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            False, False, 0.5, 0.5, 0.5, 0.0)
        np.testing.assert_array_equal(buffer, original)

    def test_zero_size(self):
        buffer = np.array([100, 100, 100, 255], dtype=np.uint8)
        original = buffer.copy()
        _apply_adjustments_fast(buffer, 0, 0, 4,
            0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            False, False, 0.5, 0.5, 0.5, 0.0)
        np.testing.assert_array_equal(buffer, original)
```

### 10.2 Regression Tests (run during migration before deleting Numba code)

Compare C++ output vs Numba output across random pixel buffers to ensure identical results.

### 10.3 Performance Benchmark

Expected: C++ should be 5-20x faster than NumPy for large images, comparable to Numba AOT.

---

## 11. Fallback and Compatibility

### 11.1 Graceful Degradation

    _cpp_filters (C++ via pybind11) -> (ImportError) -> numpy_executor.py (pure NumPy)

### 11.2 Environment Compatibility

| Environment | C++ Extension | NumPy Fallback |
|------------|--------------|----------------|
| Development (pip install -e .) | Yes (if built) | Yes |
| Production (pip install .) | Yes | Yes |
| Nuitka package | Yes | Yes |
| PyInstaller | Yes | Yes |
| No C++ compiler | No | Yes |

### 11.3 Platform Support

| Platform | Compiler |
|----------|----------|
| Linux (x86_64) | GCC 9+ / Clang 10+ |
| macOS (arm64/x86_64) | Apple Clang 13+ |
| Windows (x86_64) | MSVC 2019+ |

---

## 12. Dependency Changes

### 12.1 Dependencies to Remove

| Package | Current Version | Reason |
|---------|----------------|--------|
| numba | >=0.60 | Replaced by C++ extension |
| llvmlite | (transitive) | No longer needed without numba |

### 12.2 Dependencies to Add

| Package | Version | Type | Reason |
|---------|---------|------|--------|
| pybind11 | >=2.12 | Build-time only | C++ binding generation |
| cmake | >=3.15 | Build-time only | C++ build system |

---

## 13. FAQ

**Q: Will the NumPy fallback path still work?**
A: Yes. numpy_executor.py is completely unchanged.

**Q: Can I still develop without building the C++ extension?**
A: Yes. The app automatically falls back to NumPy.

**Q: How does this affect the Nuitka build process?**
A: It simplifies. Build the pybind11 extension with CMake instead of running build_jit.py.

**Q: What C++ standard is required?**
A: C++17 for structured bindings.

**Q: Can OpenMP be used for parallelization?**
A: Yes. CMakeLists.txt includes optional OpenMP support. Benchmark before enabling.

**Q: What about demo/curve/curve.py?**
A: Remove numba import, keep the fallback decorator. Demo runs as plain Python.

---

## Summary of All Changes

### Files to CREATE (7 files)

| File | Purpose |
|------|---------|
| src/iPhoto/core/filters/cpp/CMakeLists.txt | CMake build config |
| src/iPhoto/core/filters/cpp/src/algorithms.h | C++ algorithm headers |
| src/iPhoto/core/filters/cpp/src/algorithms.cpp | C++ algorithm implementations |
| src/iPhoto/core/filters/cpp/src/kernels.h | C++ kernel headers |
| src/iPhoto/core/filters/cpp/src/kernels.cpp | C++ kernel implementations |
| src/iPhoto/core/filters/cpp/src/bindings.cpp | pybind11 module definition |
| setup.py (project root) | CMake extension build integration |

### Files to MODIFY (5 files)

| File | Change |
|------|--------|
| src/iPhoto/core/filters/jit_executor.py | Replace AOT/JIT with _cpp_filters import |
| src/iPhoto/core/filters/algorithms.py | Remove @jit decorators and numba import |
| src/iPhoto/core/color_resolver.py | Remove @jit decorator and numba import |
| demo/curve/curve.py | Remove numba import |
| pyproject.toml | Remove numba dep; add pybind11 build dep |

### Files to DELETE (2 files)

| File | Reason |
|------|--------|
| src/iPhoto/core/filters/jit_kernels.py | Replaced by cpp/src/kernels.cpp |
| src/iPhoto/core/filters/build_jit.py | Replaced by cpp/CMakeLists.txt |

### Documentation to UPDATE (1 file)

| File | Change |
|------|--------|
| docs/BUILD_EXE.md | Replace Numba AOT instructions with C++ build instructions |
