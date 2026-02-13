"""Shader source loading helpers for GL widgets."""

from __future__ import annotations

from pathlib import Path


def load_shader_source(module_file: str, filename: str) -> str:
    """Return GLSL source stored next to *module_file*."""

    shader_path = Path(module_file).resolve().with_name(filename)
    try:
        return shader_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to load shader '{filename}': {exc}") from exc

