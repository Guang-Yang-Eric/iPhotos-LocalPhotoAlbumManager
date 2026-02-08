# -*- coding: utf-8 -*-
"""Tests for the Live Photo edit controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.iPhoto.gui.ui.controllers.live_photo_edit_controller import (
    LivePhotoEditController,
)


@dataclass
class _FakeLiveGroup:
    id: str = "live_test"
    still: str = "/photos/IMG_001.HEIC"
    motion: str = "/photos/IMG_001.mov"
    content_id: Optional[str] = None
    still_image_time: Optional[float] = None
    confidence: float = 1.0


class TestLivePhotoEditController:
    """Unit tests for LivePhotoEditController mode switching."""

    def _make_controller(self):
        """Create a LivePhotoEditController with mocked internals."""
        ctrl = LivePhotoEditController.__new__(LivePhotoEditController)
        ctrl._live_group = _FakeLiveGroup()
        ctrl._gl_viewer = MagicMock()
        ctrl._adjustments = {}
        ctrl._still_image = None
        ctrl._player = None
        ctrl._audio = None
        ctrl._video_coordinator = None
        ctrl._mode = "still"
        ctrl.modeChanged = MagicMock()
        return ctrl

    def test_default_mode_is_still(self):
        """Controller should start in 'still' mode."""
        ctrl = self._make_controller()
        assert ctrl.mode == "still"

    def test_toggle_mode_switches(self):
        """toggle_mode() should alternate between still and motion."""
        ctrl = self._make_controller()

        # still → motion
        ctrl.toggle_mode()
        assert ctrl._mode == "motion"

        # motion → still
        ctrl.toggle_mode()
        assert ctrl._mode == "still"

    def test_update_adjustments_propagates_to_coordinator(self):
        """Adjustments should be forwarded to the video coordinator when present."""
        ctrl = self._make_controller()
        mock_coordinator = MagicMock()
        ctrl._video_coordinator = mock_coordinator

        adj = {"Exposure": 0.5}
        ctrl.update_adjustments(adj)

        assert ctrl._adjustments == adj
        mock_coordinator.update_adjustments.assert_called_once_with(adj)

    def test_enter_still_mode_loads_image(self):
        """enter_still_mode should upload the still texture to the GL viewer."""
        ctrl = self._make_controller()

        # Pre-populate still image to avoid QImage constructor call
        mock_still = MagicMock()
        ctrl._still_image = mock_still

        ctrl.enter_still_mode()

        assert ctrl._mode == "still"
        ctrl._gl_viewer.renderer.upload_texture.assert_called_once_with(mock_still)
        ctrl._gl_viewer.set_adjustments.assert_called_once()
        ctrl._gl_viewer.update.assert_called_once()
