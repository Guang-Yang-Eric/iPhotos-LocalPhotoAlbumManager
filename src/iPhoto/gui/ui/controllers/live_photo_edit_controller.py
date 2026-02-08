# -*- coding: utf-8 -*-
"""Controller for Live Photo editing (still + motion dual-mode).

A Live Photo is a pair of:
  * **still** – high-quality JPEG/HEIC (edited like a regular photo)
  * **motion** – short .mov clip (1-3 s, same adjustments applied per-frame)

Both modes share a single EditSession so adjustments made on the still image
automatically apply when the motion component is previewed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional

from PySide6.QtCore import QObject, Signal

from .video_edit_coordinator import VideoEditCoordinator

if TYPE_CHECKING:
    from PySide6.QtMultimedia import QMediaPlayer
    from ....models.types import LiveGroup
    from ..widgets.gl_image_viewer import GLImageViewer

_LOGGER = logging.getLogger(__name__)


class LivePhotoEditController(QObject):
    """Manage dual-mode editing for a Live Photo asset.

    Modes:
      * **still** – standard photo editing pipeline (GLRenderer + static QImage)
      * **motion** – ``VideoEditCoordinator`` renders colour-graded video frames

    Both modes share the same ``EditSession`` (adjustments dict), ensuring
    visual consistency between the still cover and the motion clip.
    """

    modeChanged = Signal(str)
    """Emitted with ``"still"`` or ``"motion"`` when the active mode switches."""

    def __init__(
        self,
        live_group: LiveGroup,
        gl_viewer: GLImageViewer,
        initial_adjustments: Mapping[str, object] | None = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._live_group = live_group
        self._gl_viewer = gl_viewer
        self._adjustments: dict[str, object] = dict(initial_adjustments or {})

        # Still image (lazy-loaded)
        self._still_image = None  # Optional[QImage]

        # Motion player (lazy-initialised)
        self._player: Optional[QMediaPlayer] = None
        self._audio = None  # Optional[QAudioOutput]
        self._video_coordinator: Optional[VideoEditCoordinator] = None

        self._mode = "still"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        """Return the current editing mode: ``"still"`` or ``"motion"``."""
        return self._mode

    def update_adjustments(self, adjustments: Mapping[str, object]) -> None:
        """Push updated adjustments to the active rendering pipeline."""
        self._adjustments = dict(adjustments)
        if self._video_coordinator is not None:
            self._video_coordinator.update_adjustments(self._adjustments)

    def enter_still_mode(self) -> None:
        """Switch to the static photo editing pipeline."""
        self._mode = "still"

        # Pause any playing motion clip
        if self._player is not None:
            try:
                from PySide6.QtMultimedia import QMediaPlayer

                if (
                    self._player.playbackState()
                    == QMediaPlayer.PlaybackState.PlayingState
                ):
                    self._player.pause()
            except (ImportError, RuntimeError):
                pass

        # Load the still if not cached
        if self._still_image is None:
            from PySide6.QtGui import QImage

            still_path = str(self._live_group.still)
            self._still_image = QImage(still_path)

        renderer = self._gl_viewer.renderer
        if renderer is not None and self._still_image is not None:
            renderer.upload_texture(self._still_image)
            self._gl_viewer.set_adjustments(self._adjustments)
            self._gl_viewer.update()

        self.modeChanged.emit("still")

    def enter_motion_mode(self) -> None:
        """Switch to the motion video preview pipeline."""
        self._mode = "motion"
        self._ensure_player()

        if self._video_coordinator is None and self._player is not None:
            self._video_coordinator = VideoEditCoordinator(
                self._player,
                self._gl_viewer,
                initial_adjustments=self._adjustments,
                parent=self,
            )

        if self._player is not None:
            from PySide6.QtCore import QUrl

            motion_path = str(self._live_group.motion)
            self._player.setSource(QUrl.fromLocalFile(motion_path))
            self._player.play()

        self.modeChanged.emit("motion")

    def toggle_mode(self) -> None:
        """Toggle between still and motion editing modes."""
        if self._mode == "still":
            self.enter_motion_mode()
        else:
            self.enter_still_mode()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_player(self) -> None:
        """Lazily create the QMediaPlayer and QAudioOutput."""
        if self._player is not None:
            return
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self._player = QMediaPlayer(self)
            self._audio = QAudioOutput(self)
            self._player.setAudioOutput(self._audio)
        except (ImportError, RuntimeError):
            _LOGGER.warning("Qt Multimedia not available; motion preview disabled.")
