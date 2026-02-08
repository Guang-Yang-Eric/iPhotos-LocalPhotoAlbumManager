# -*- coding: utf-8 -*-
"""Tests for the GLRenderer.upload_texture_incremental method."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestUploadTextureIncremental:
    """Unit tests for GLRenderer.upload_texture_incremental."""

    def _make_renderer(self):
        """Create a minimal GLRenderer with mocked GL functions."""
        from src.iPhoto.gui.ui.widgets.gl_renderer import GLRenderer

        mock_gl = MagicMock()
        renderer = GLRenderer.__new__(GLRenderer)
        renderer._gl_funcs = mock_gl
        renderer._parent = None
        renderer._program = None
        renderer._dummy_vao = None
        renderer._uniform_locations = {}
        renderer._texture_id = 0
        renderer._texture_width = 0
        renderer._texture_height = 0
        renderer._curve_lut_texture_id = 0
        renderer._levels_lut_texture_id = 0
        renderer._overlay_program = None
        renderer._overlay_vao = None
        renderer._overlay_vbo = 0
        return renderer

    @patch("src.iPhoto.gui.ui.widgets.gl_renderer.gl")
    def test_incremental_when_same_size(self, mock_gl):
        """When dimensions match, should use glTexSubImage2D (return True)."""
        renderer = self._make_renderer()
        renderer._texture_id = 42
        renderer._texture_width = 100
        renderer._texture_height = 80

        mock_image = MagicMock()
        mock_image.width.return_value = 100
        mock_image.height.return_value = 80
        mock_converted = MagicMock()
        mock_converted.constBits.return_value = b"\x00" * (100 * 80 * 4)
        mock_converted.sizeInBytes.return_value = 100 * 80 * 4
        mock_converted.bytesPerLine.return_value = 100 * 4
        mock_image.convertToFormat.return_value = mock_converted

        result = renderer.upload_texture_incremental(mock_image)

        assert result is True
        mock_gl.glTexSubImage2D.assert_called_once()
        # Should NOT create a new texture
        mock_gl.glGenTextures.assert_not_called()

    @patch("src.iPhoto.gui.ui.widgets.gl_renderer.gl")
    def test_full_rebuild_when_size_differs(self, mock_gl):
        """When dimensions differ, should rebuild (return False)."""
        renderer = self._make_renderer()
        renderer._texture_id = 42
        renderer._texture_width = 100
        renderer._texture_height = 80

        mock_image = MagicMock()
        mock_image.width.return_value = 200  # Different width
        mock_image.height.return_value = 160  # Different height
        mock_image.isNull.return_value = False
        mock_converted = MagicMock()
        mock_converted.constBits.return_value = b"\x00" * (200 * 160 * 4)
        mock_converted.sizeInBytes.return_value = 200 * 160 * 4
        mock_converted.bytesPerLine.return_value = 200 * 4
        mock_image.convertToFormat.return_value = mock_converted

        mock_gl.glGenTextures.return_value = 99
        mock_gl.GL_NO_ERROR = 0
        mock_gl.glGetError.return_value = 0

        result = renderer.upload_texture_incremental(mock_image)

        assert result is False

    @patch("src.iPhoto.gui.ui.widgets.gl_renderer.gl")
    def test_full_rebuild_when_no_texture(self, mock_gl):
        """When no texture exists, should do full upload (return False)."""
        renderer = self._make_renderer()
        renderer._texture_id = 0  # No texture

        mock_image = MagicMock()
        mock_image.width.return_value = 100
        mock_image.height.return_value = 80
        mock_image.isNull.return_value = False
        mock_converted = MagicMock()
        mock_converted.constBits.return_value = b"\x00" * (100 * 80 * 4)
        mock_converted.sizeInBytes.return_value = 100 * 80 * 4
        mock_converted.bytesPerLine.return_value = 100 * 4
        mock_image.convertToFormat.return_value = mock_converted

        mock_gl.glGenTextures.return_value = 1
        mock_gl.GL_NO_ERROR = 0
        mock_gl.glGetError.return_value = 0

        result = renderer.upload_texture_incremental(mock_image)

        assert result is False
