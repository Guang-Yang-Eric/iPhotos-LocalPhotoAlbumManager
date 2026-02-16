# -*- coding: utf-8 -*-
"""Shader compilation and program management for the GL renderer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtOpenGL import (
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from OpenGL import GL as gl

if TYPE_CHECKING:
    from iPhoto.infrastructure.services.gpu_pipeline import ShaderPrecompiler

_LOGGER = logging.getLogger(__name__)


_OVERLAY_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""


_OVERLAY_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main() {
    FragColor = uColor;
}
"""


_UNIFORM_NAMES = (
    "uTex",
    "uBrilliance",
    "uExposure",
    "uHighlights",
    "uShadows",
    "uBrightness",
    "uContrast",
    "uBlackPoint",
    "uSaturation",
    "uVibrance",
    "uColorCast",
    "uGain",
    "uBWParams",
    "uBWEnabled",
    "uCurveLUT",
    "uCurveEnabled",
    "uLevelsLUT",
    "uLevelsEnabled",
    "uWBWarmth",
    "uWBTemperature",
    "uWBTint",
    "uWBEnabled",
    "uTime",
    "uViewSize",
    "uTexSize",
    "uScale",
    "uPan",
    "uImgScale",
    "uImgOffset",
    "uCropCX",
    "uCropCY",
    "uCropW",
    "uCropH",
    "uPerspectiveMatrix",
    "uRotate90",
    "uSCRange0",
    "uSCRange1",
    "uSCEnabled",
)


def _load_shader_source(filename: str) -> str:
    """Return the GLSL source stored alongside this module."""

    shader_path = Path(__file__).resolve().with_name(filename)
    try:
        return shader_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to load shader '{filename}': {exc}") from exc


class ShaderManager:
    """Owns the main and overlay shader programs, VAOs, and uniform location cache."""

    def __init__(
        self,
        gl_funcs,
        *,
        parent: Optional[QObject] = None,
        shader_precompiler: Optional[ShaderPrecompiler] = None,
    ) -> None:
        self._gl_funcs = gl_funcs
        self._parent = parent
        self._precompiler = shader_precompiler
        self.program: Optional[QOpenGLShaderProgram] = None
        self.dummy_vao: Optional[QOpenGLVertexArrayObject] = None
        self.uniform_locations: dict[str, int] = {}
        self.overlay_program: Optional[QOpenGLShaderProgram] = None
        self.overlay_vao: Optional[QOpenGLVertexArrayObject] = None
        self.overlay_vbo: int = 0

    def initialize(self) -> None:
        """Compile shaders, create VAOs, and cache uniform locations."""

        self.destroy()

        if self._precompiler is not None:
            _LOGGER.info("GPU Pipeline: shader compilation via ShaderPrecompiler (batch)")
            self._compile_shaders_precompiled()
        else:
            _LOGGER.info("GPU Pipeline: shader compilation via JIT (direct Qt API)")
            self._compile_shaders_jit()

        # Shared setup: VAO, GL state, uniform locations, overlay VBO
        vao = QOpenGLVertexArrayObject(self._parent)
        vao.create()
        self.dummy_vao = vao if vao.isCreated() else None

        gf = self._gl_funcs
        gf.glDisable(gl.GL_DEPTH_TEST)
        gf.glDisable(gl.GL_CULL_FACE)
        gf.glDisable(gl.GL_BLEND)

        self.program.bind()
        try:
            for name in _UNIFORM_NAMES:
                self.uniform_locations[name] = self.program.uniformLocation(name)
        finally:
            self.program.release()

        overlay_vao = QOpenGLVertexArrayObject(self._parent)
        overlay_vao.create()
        self.overlay_vao = overlay_vao if overlay_vao.isCreated() else None
        buffer_id = gl.glGenBuffers(1)
        if isinstance(buffer_id, (tuple, list)):
            buffer_id = buffer_id[0]
        self.overlay_vbo = int(buffer_id)

    # ------------------------------------------------------------------
    # Precompiled shader path (GPU Pipeline integration)
    # ------------------------------------------------------------------
    def _compile_shaders_precompiled(self) -> None:
        """Batch-compile all shaders via :class:`ShaderPrecompiler`."""
        from iPhoto.infrastructure.services.gpu_pipeline import ShaderSource

        vert = _load_shader_source("gl_image_viewer.vert")
        frag = _load_shader_source("gl_image_viewer.frag")
        self._precompiler.register(ShaderSource("main", vert, frag))
        self._precompiler.register(
            ShaderSource("overlay", _OVERLAY_VERTEX_SHADER, _OVERLAY_FRAGMENT_SHADER)
        )

        self._precompiler.compile_all()

        main = self._precompiler.get("main")
        if not main or not main.success:
            msg = main.error if main else "shader not found"
            _LOGGER.error("Main shader precompilation failed: %s", msg)
            raise RuntimeError(f"Main shader precompilation failed: {msg}")
        self.program = main.program

        overlay = self._precompiler.get("overlay")
        if not overlay or not overlay.success:
            msg = overlay.error if overlay else "shader not found"
            _LOGGER.error("Overlay shader precompilation failed: %s", msg)
            raise RuntimeError(f"Overlay shader precompilation failed: {msg}")
        self.overlay_program = overlay.program

    @staticmethod
    def create_qt_compile_fn(parent: Optional[QObject] = None):
        """Return a :data:`CompileFn` that wraps Qt shader compilation.

        The returned function can be passed to
        :class:`~iPhoto.infrastructure.services.gpu_pipeline.ShaderPrecompiler`
        so that all registered shaders are compiled via the Qt OpenGL API.
        """
        from iPhoto.infrastructure.services.gpu_pipeline import CompiledShader

        def _compile(source):
            prog = QOpenGLShaderProgram(parent)
            if not prog.addShaderFromSourceCode(QOpenGLShader.Vertex, source.vertex_source):
                return CompiledShader(
                    name=source.name, program=None, success=False, error=prog.log()
                )
            if not prog.addShaderFromSourceCode(QOpenGLShader.Fragment, source.fragment_source):
                return CompiledShader(
                    name=source.name, program=None, success=False, error=prog.log()
                )
            if not prog.link():
                return CompiledShader(
                    name=source.name, program=None, success=False, error=prog.log()
                )
            return CompiledShader(name=source.name, program=prog, success=True)

        return _compile

    # ------------------------------------------------------------------
    # JIT shader path (original fallback)
    # ------------------------------------------------------------------
    def _compile_shaders_jit(self) -> None:
        """Compile shaders directly via Qt API (JIT fallback)."""
        program = QOpenGLShaderProgram(self._parent)
        vert_source = _load_shader_source("gl_image_viewer.vert")
        frag_source = _load_shader_source("gl_image_viewer.frag")
        if not program.addShaderFromSourceCode(QOpenGLShader.Vertex, vert_source):
            message = program.log()
            _LOGGER.error("Vertex shader compilation failed: %s", message)
            raise RuntimeError("Unable to compile vertex shader")
        if not program.addShaderFromSourceCode(QOpenGLShader.Fragment, frag_source):
            message = program.log()
            _LOGGER.error("Fragment shader compilation failed: %s", message)
            raise RuntimeError("Unable to compile fragment shader")
        if not program.link():
            message = program.log()
            _LOGGER.error("Shader program link failed: %s", message)
            raise RuntimeError("Unable to link shader program")
        self.program = program

        overlay_prog = QOpenGLShaderProgram(self._parent)
        if not overlay_prog.addShaderFromSourceCode(
            QOpenGLShader.Vertex, _OVERLAY_VERTEX_SHADER
        ):
            raise RuntimeError("Unable to compile overlay vertex shader")
        if not overlay_prog.addShaderFromSourceCode(
            QOpenGLShader.Fragment, _OVERLAY_FRAGMENT_SHADER
        ):
            raise RuntimeError("Unable to compile overlay fragment shader")
        if not overlay_prog.link():
            raise RuntimeError("Unable to link overlay shader program")
        self.overlay_program = overlay_prog

    def destroy(self) -> None:
        """Release shader programs, VAOs, and the overlay VBO."""

        if self.dummy_vao is not None:
            self.dummy_vao.destroy()
            self.dummy_vao = None
        if self.program is not None:
            self.program.removeAllShaders()
            self.program = None
        self.uniform_locations.clear()
        if self.overlay_vao is not None:
            self.overlay_vao.destroy()
            self.overlay_vao = None
        if self.overlay_program is not None:
            self.overlay_program.removeAllShaders()
            self.overlay_program = None
        if self.overlay_vbo:
            gl.glDeleteBuffers(1, np.array([int(self.overlay_vbo)], dtype=np.uint32))
            self.overlay_vbo = 0
