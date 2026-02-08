# -*- coding: utf-8 -*-
"""Orchestrate the video editing pipeline.

Coordinates the flow:
  QMediaPlayer → VideoFrameProcessor → GLRenderer → GLImageViewer

The coordinator listens to ``EditSession.valuesChanged`` and caches the latest
adjustments dict.  Each incoming video frame is rendered with the most recent
parameters so slider drags take effect within a single frame period (~33 ms at
30 fps).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping, Optional

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from .video_frame_processor import VideoFrameProcessor

if TYPE_CHECKING:
    from PySide6.QtMultimedia import QMediaPlayer
    from ..widgets.gl_image_viewer import GLImageViewer

_LOGGER = logging.getLogger(__name__)


class VideoEditCoordinator(QObject):
    """Wire *player* frames through a GL colour-grading pipeline.

    Parameters
    ----------
    player:
        The ``QMediaPlayer`` instance that decodes the video.
    gl_viewer:
        The ``GLImageViewer`` widget that displays the graded output.
    edit_session_values:
        A callable returning the current adjustments dict from EditSession.
    parent:
        Optional parent QObject.
    """

    renderCompleted = Signal()
    """Emitted after every successful frame render."""

    def __init__(
        self,
        player: QMediaPlayer,
        gl_viewer: GLImageViewer,
        initial_adjustments: Mapping[str, object] | None = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._player = player
        self._gl_viewer = gl_viewer
        self._frame_processor = VideoFrameProcessor(self)

        # Redirect player output to our sink
        self._player.setVideoOutput(self._frame_processor.video_sink)

        # Frame → render
        self._frame_processor.frameReady.connect(self._render_frame)

        # Adjustments cache
        self._current_adjustments: dict[str, object] = dict(initial_adjustments or {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def frame_processor(self) -> VideoFrameProcessor:
        """Expose the processor for external wiring."""
        return self._frame_processor

    def update_adjustments(self, adjustments: Mapping[str, object]) -> None:
        """Cache the latest adjustments dict from EditSession.

        This is a lightweight dict-copy operation (< 1 ms); actual rendering
        happens on the next ``videoFrameChanged`` signal.

        If the video is paused, an immediate ``update()`` is triggered so the
        user sees the adjustment take effect on the current still frame.
        """
        self._current_adjustments = dict(adjustments)

        # When paused, re-render the frozen frame with updated params
        try:
            from PySide6.QtMultimedia import QMediaPlayer

            if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self._gl_viewer.update()
        except (ImportError, RuntimeError):
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @Slot(QImage)
    def _render_frame(self, frame_image: QImage) -> None:
        """Upload a new video frame texture and trigger GL repaint."""
        renderer = self._gl_viewer.renderer
        if renderer is None:
            self._frame_processor.mark_render_complete()
            return

        try:
            # Incremental texture upload (fast path when size unchanged)
            renderer.upload_texture_incremental(frame_image)

            # Update adjustments and repaint
            self._gl_viewer.set_adjustments(self._current_adjustments)
            self._gl_viewer.update()
        except Exception:
            _LOGGER.debug("Frame render failed", exc_info=True)
        finally:
            self._frame_processor.mark_render_complete()

        self.renderCompleted.emit()
