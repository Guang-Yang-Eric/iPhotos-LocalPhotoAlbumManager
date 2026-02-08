# -*- coding: utf-8 -*-
"""Tests for the video export pipeline core module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetVideoMetadata:
    """Tests for get_video_metadata."""

    def test_returns_defaults_when_av_missing(self, monkeypatch):
        """Should return safe defaults when PyAV is not installed."""
        import importlib
        import src.iPhoto.core.video_export_pipeline as pipeline

        with patch.dict("sys.modules", {"av": None}):
            # Re-run with av import failing
            result = pipeline.get_video_metadata(Path("nonexistent.mp4"))
            assert result["fps"] == 30.0
            assert result["width"] == 0

    def test_returns_metadata_from_container(self, monkeypatch):
        """Should extract fps, width, height from a valid container."""
        import src.iPhoto.core.video_export_pipeline as pipeline

        mock_av = MagicMock()
        mock_stream = MagicMock()
        mock_stream.average_rate = 24
        mock_stream.duration = 1000
        mock_stream.time_base = 0.001  # 1 ms
        mock_stream.width = 1920
        mock_stream.height = 1080

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)
        mock_av.open.return_value = mock_container

        with patch.object(pipeline, "av", mock_av, create=True):
            with patch.dict("sys.modules", {"av": mock_av}):
                result = pipeline.get_video_metadata(Path("test.mp4"))

        assert result["fps"] == 24.0
        assert result["width"] == 1920
        assert result["height"] == 1080


class TestIterateDecodedFrames:
    """Tests for iterate_decoded_frames."""

    def test_yields_frames(self):
        """Should yield PIL images from the decoded video stream."""
        import src.iPhoto.core.video_export_pipeline as pipeline

        mock_av = MagicMock()
        mock_frame = MagicMock()
        mock_image = MagicMock()
        mock_image.size = (1920, 1080)
        mock_frame.to_image.return_value = mock_image

        mock_stream = MagicMock()
        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = [mock_frame]
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)
        mock_av.open.return_value = mock_container

        with patch.dict("sys.modules", {"av": mock_av}):
            frames = list(pipeline.iterate_decoded_frames(Path("test.mp4")))

        assert len(frames) == 1
        assert frames[0] is mock_image

    def test_empty_video(self):
        """Should yield nothing for a video with no video streams."""
        import src.iPhoto.core.video_export_pipeline as pipeline

        mock_av = MagicMock()
        mock_container = MagicMock()
        mock_container.streams.video = []
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)
        mock_av.open.return_value = mock_container

        with patch.dict("sys.modules", {"av": mock_av}):
            frames = list(pipeline.iterate_decoded_frames(Path("empty.mp4")))

        assert frames == []
