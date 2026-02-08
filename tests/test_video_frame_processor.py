# -*- coding: utf-8 -*-
"""Tests for the video frame processor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.iPhoto.gui.ui.controllers.video_frame_processor import VideoFrameProcessor


class TestVideoFrameProcessor:
    """Unit tests for VideoFrameProcessor frame interception logic."""

    def _make_processor(self):
        """Create a VideoFrameProcessor with mocked internals."""
        proc = VideoFrameProcessor.__new__(VideoFrameProcessor)
        proc._rendering = False
        proc._last_size = (0, 0)
        proc._sink = MagicMock()
        proc.frameReady = MagicMock()
        return proc

    def test_rendering_lock_drops_frame(self):
        """When _rendering is True, incoming frames must be silently dropped."""
        proc = self._make_processor()
        proc._rendering = True

        mock_frame = MagicMock()
        mock_frame.isValid.return_value = True

        proc._on_frame(mock_frame)
        mock_frame.toImage.assert_not_called()

    def test_invalid_frame_is_skipped(self):
        """Invalid QVideoFrame should not trigger any conversion."""
        proc = self._make_processor()

        mock_frame = MagicMock()
        mock_frame.isValid.return_value = False

        proc._on_frame(mock_frame)
        mock_frame.toImage.assert_not_called()

    def test_null_image_is_skipped(self):
        """Null QImage from toImage() should not emit frameReady."""
        proc = self._make_processor()

        mock_frame = MagicMock()
        mock_frame.isValid.return_value = True
        mock_image = MagicMock()
        mock_image.isNull.return_value = True
        mock_frame.toImage.return_value = mock_image

        proc._on_frame(mock_frame)
        proc.frameReady.emit.assert_not_called()

    def test_mark_render_complete_releases_lock(self):
        """mark_render_complete() must set _rendering to False."""
        proc = self._make_processor()
        proc._rendering = True

        proc.mark_render_complete()
        assert proc._rendering is False

    def test_last_frame_size_default(self):
        """Default last_frame_size should be (0, 0)."""
        proc = self._make_processor()
        assert proc.last_frame_size == (0, 0)
