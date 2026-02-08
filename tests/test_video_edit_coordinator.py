# -*- coding: utf-8 -*-
"""Tests for the video edit coordinator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.iPhoto.gui.ui.controllers.video_edit_coordinator import VideoEditCoordinator


class TestVideoEditCoordinator:
    """Unit tests for VideoEditCoordinator adjustment caching."""

    def _make_coordinator(self):
        """Create a VideoEditCoordinator with mocked internals."""
        coord = VideoEditCoordinator.__new__(VideoEditCoordinator)
        coord._current_adjustments = {}
        coord._player = MagicMock()
        coord._gl_viewer = MagicMock()
        coord._frame_processor = MagicMock()
        coord.renderCompleted = MagicMock()
        return coord

    def test_update_adjustments_caches_dict(self):
        """update_adjustments() must cache a copy of the provided dict."""
        coord = self._make_coordinator()

        # Simulate playing state so update() is not called
        coord._player.playbackState.return_value = "playing"

        new_adj = {"Exposure": 0.5, "Contrast": -0.2}
        coord.update_adjustments(new_adj)

        assert coord._current_adjustments == new_adj
        assert coord._current_adjustments is not new_adj

    def test_update_adjustments_triggers_update_when_paused(self):
        """When the player is paused, update_adjustments triggers gl_viewer.update()."""
        coord = self._make_coordinator()
        coord._player.playbackState.return_value = "paused"

        coord.update_adjustments({"Exposure": 0.3})
        coord._gl_viewer.update.assert_called()

    def test_render_frame_uploads_and_repaints(self):
        """_render_frame should call upload_texture_incremental and update."""
        coord = self._make_coordinator()
        coord._current_adjustments = {"Exposure": 0.0}

        mock_renderer = MagicMock()
        coord._gl_viewer.renderer = mock_renderer

        mock_image = MagicMock()
        coord._render_frame(mock_image)

        mock_renderer.upload_texture_incremental.assert_called_once_with(mock_image)
        coord._gl_viewer.set_adjustments.assert_called_once()
        coord._gl_viewer.update.assert_called_once()
        coord._frame_processor.mark_render_complete.assert_called_once()

    def test_render_frame_handles_no_renderer(self):
        """_render_frame must not crash when renderer is None."""
        coord = self._make_coordinator()
        coord._gl_viewer.renderer = None

        mock_image = MagicMock()
        coord._render_frame(mock_image)

        coord._frame_processor.mark_render_complete.assert_called_once()
