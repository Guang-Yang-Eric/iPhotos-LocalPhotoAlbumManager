# -*- coding: utf-8 -*-
"""Tests for qimage_to_numpy and encode_video_from_frames in ffmpeg.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.iPhoto.utils import ffmpeg
from src.iPhoto.errors import ExternalToolError


class _FakeQImage:
    """Minimal QImage stand-in for testing qimage_to_numpy."""

    class Format:
        Format_RGB888 = "rgb888"
        Format_RGBA8888 = "rgba8888"

    def __init__(self, data: bytes, w: int, h: int, channels: int = 3):
        self._data = data
        self._w = w
        self._h = h
        self._channels = channels
        if channels == 3:
            self._format = self.Format.Format_RGB888
        else:
            self._format = self.Format.Format_RGBA8888

    def format(self):
        return self._format

    def width(self):
        return self._w

    def height(self):
        return self._h

    def sizeInBytes(self):
        return len(self._data)

    def bytesPerLine(self):
        return self._w * self._channels

    def constBits(self):
        return self._data

    def convertToFormat(self, fmt):
        return self  # Already in the right format for testing


class TestQImageToNumpy:
    """Unit tests for the qimage_to_numpy helper."""

    def test_rgb888_image(self, monkeypatch):
        """An RGB888 QImage should produce an (H, W, 3) array."""
        # 2x2 RGB image
        raw = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128])
        img = _FakeQImage(raw, 2, 2, 3)

        monkeypatch.setattr(
            "src.iPhoto.utils.ffmpeg.QImage",
            type("FakeQImageModule", (), {"Format": _FakeQImage.Format}),
            raising=False,
        )

        # Patch the module-level reference used by qimage_to_numpy
        with patch("src.iPhoto.utils.ffmpeg.qimage_to_numpy") as mock_fn:
            # Instead of patching, call the real function with our fake
            pass

        # Call directly with our fake QImage
        with patch("PySide6.QtGui.QImage", _FakeQImage):
            arr = ffmpeg.qimage_to_numpy(img)

        assert arr.shape == (2, 2, 3)
        assert arr.dtype == np.uint8
        assert arr[0, 0, 0] == 255  # R
        assert arr[0, 1, 1] == 255  # G

    def test_rgba8888_drops_alpha(self, monkeypatch):
        """An RGBA8888 QImage should produce an (H, W, 3) array with alpha dropped."""
        # 1x2 RGBA image
        raw = bytes([255, 0, 0, 255, 0, 255, 0, 128])
        img = _FakeQImage(raw, 2, 1, 4)

        with patch("PySide6.QtGui.QImage", _FakeQImage):
            arr = ffmpeg.qimage_to_numpy(img)

        assert arr.shape == (1, 2, 3)
        assert arr[0, 0, 0] == 255  # R
        assert arr[0, 1, 1] == 255  # G


class TestEncodeVideoFromFrames:
    """Unit tests for encode_video_from_frames."""

    def test_raises_when_pyav_unavailable(self, monkeypatch):
        """Should raise ExternalToolError when av is None."""
        monkeypatch.setattr(ffmpeg, "av", None)

        with pytest.raises(ExternalToolError, match="PyAV is required"):
            ffmpeg.encode_video_from_frames(
                Path("/tmp/out.mp4"),
                iter([]),
                fps=30.0,
            )

    def test_encodes_frames_successfully(self, monkeypatch, tmp_path):
        """Verify the encoding pipeline opens, muxes, and closes containers."""
        mock_av = MagicMock()
        monkeypatch.setattr(ffmpeg, "av", mock_av)

        mock_output = MagicMock()
        mock_av.open.return_value = mock_output

        mock_video_stream = MagicMock()
        mock_video_stream.width = 0
        mock_video_stream.height = 0
        mock_output.add_stream.return_value = mock_video_stream

        # Encode returns empty list (no packets for simplicity)
        mock_video_stream.encode.return_value = []

        # Create a fake QImage-like frame
        fake_img = _FakeQImage(bytes(6), 2, 1, 3)

        out = tmp_path / "test.mp4"
        ffmpeg.encode_video_from_frames(out, [fake_img], fps=30.0)

        mock_av.open.assert_called_once_with(str(out), mode="w")
        assert mock_video_stream.encode.called
        mock_output.close.assert_called_once()

    def test_audio_stream_copy(self, monkeypatch, tmp_path):
        """When audio_source is provided, audio stream should be copied."""
        mock_av = MagicMock()
        monkeypatch.setattr(ffmpeg, "av", mock_av)

        # Output container
        mock_output = MagicMock()
        mock_video_stream = MagicMock()
        mock_video_stream.width = 0
        mock_video_stream.height = 0
        mock_video_stream.encode.return_value = []
        mock_output.add_stream.return_value = mock_video_stream

        # Audio input container
        mock_audio_input = MagicMock()
        mock_audio_stream = MagicMock()
        mock_audio_input.streams.audio = [mock_audio_stream]

        # Demux returns one packet
        mock_packet = MagicMock()
        mock_packet.dts = 100
        mock_audio_input.demux.return_value = [mock_packet]

        # open() returns different containers based on mode
        def open_side_effect(path, mode="r"):
            if mode == "w":
                return mock_output
            return mock_audio_input

        mock_av.open.side_effect = open_side_effect

        audio_src = tmp_path / "source.mp4"
        audio_src.touch()
        out = tmp_path / "output.mp4"

        ffmpeg.encode_video_from_frames(
            out,
            iter([]),  # No video frames
            fps=30.0,
            audio_source=audio_src,
        )

        mock_audio_input.close.assert_called_once()
        mock_output.close.assert_called_once()
