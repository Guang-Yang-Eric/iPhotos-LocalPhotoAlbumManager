# -*- coding: utf-8 -*-
"""Intercept video frames from QMediaPlayer via QVideoSink for GPU processing.

This module provides the frame-interception layer that converts raw video frames
into QImage objects suitable for upload to the GLRenderer texture pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot

try:  # pragma: no cover - optional Qt multimedia module
    from PySide6.QtMultimedia import QVideoFrame, QVideoSink
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    QVideoFrame = None  # type: ignore[assignment, misc]
    QVideoSink = None  # type: ignore[assignment, misc]

from PySide6.QtGui import QImage

_LOGGER = logging.getLogger(__name__)


class VideoFrameProcessor(QObject):
    """Intercept QMediaPlayer video frames and convert to QImage for GL rendering.

    Key performance points:
    - ``QVideoFrame.toImage()`` uses zero-copy mapping on PySide6 6.5+
    - Frame-drop strategy: if the GPU is still rendering the previous frame,
      the current frame is silently dropped to prevent accumulation.
    - Texture upload uses ``glTexSubImage2D`` for incremental updates
      when the frame dimensions remain unchanged.
    """

    frameReady = Signal(QImage)
    """Emitted when a converted frame image is ready for GL upload."""

    PREVIEW_MAX_DIMENSION = 1920
    """Maximum edge length for preview frames (4K → 1080p downscale)."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        if QVideoSink is None:
            raise RuntimeError(
                "PySide6.QtMultimedia is required for video frame processing."
            )
        self._sink = QVideoSink(self)
        self._sink.videoFrameChanged.connect(self._on_frame)
        self._rendering = False
        self._last_size: tuple[int, int] = (0, 0)

    @property
    def video_sink(self) -> QVideoSink:
        """Return the sink for ``QMediaPlayer.setVideoOutput()``."""
        return self._sink

    @Slot()
    def _on_frame(self, frame: QVideoFrame) -> None:
        """Process an incoming video frame."""
        # Drop frame if GPU is still rendering the previous one
        if self._rendering:
            return

        if not frame.isValid():
            return

        image = frame.toImage()
        if image.isNull():
            return

        # Convert to GL-friendly format
        if image.format() != QImage.Format.Format_RGBA8888:
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)

        # 4K → downscale to 1080p for preview (export uses original resolution)
        max_dim = max(image.width(), image.height())
        if max_dim > self.PREVIEW_MAX_DIMENSION:
            image = image.scaled(
                self.PREVIEW_MAX_DIMENSION,
                self.PREVIEW_MAX_DIMENSION,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

        self._last_size = (image.width(), image.height())
        self._rendering = True
        self.frameReady.emit(image)

    def mark_render_complete(self) -> None:
        """Signal that GL rendering is done; unblock the next frame."""
        self._rendering = False

    @property
    def last_frame_size(self) -> tuple[int, int]:
        """Return the ``(width, height)`` of the most recent processed frame."""
        return self._last_size
