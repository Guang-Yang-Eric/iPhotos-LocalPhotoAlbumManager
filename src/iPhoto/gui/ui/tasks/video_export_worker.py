# -*- coding: utf-8 -*-
"""Background worker for exporting colour-graded video files.

Runs frame-by-frame rendering on a dedicated thread so the GUI stays
responsive during long exports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from PySide6.QtCore import QObject, QThread, Signal

_LOGGER = logging.getLogger(__name__)


class VideoExportWorker(QThread):
    """Render a colour-graded video in the background.

    Signals
    -------
    progressChanged(float):
        Emitted with values in [0.0, 1.0] as frames are processed.
    exportFinished(str):
        Emitted with the output path on success.
    exportFailed(str):
        Emitted with an error message on failure.
    """

    progressChanged = Signal(float)
    exportFinished = Signal(str)
    exportFailed = Signal(str)

    def __init__(
        self,
        source: Path,
        output: Path,
        adjustments: Mapping[str, object],
        render_fn: Callable,
        *,
        codec: str = "libx264",
        quality: int = 23,
        copy_audio: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._output = output
        self._adjustments = dict(adjustments)
        self._render_fn = render_fn
        self._codec = codec
        self._quality = quality
        self._copy_audio = copy_audio
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the export."""
        self._cancelled = True

    def run(self) -> None:  # pragma: no cover - executes on worker thread
        """Perform the export (called by QThread.start())."""
        try:
            from ..core.video_export_pipeline import (
                export_graded_video,
            )

            def _progress(value: float) -> None:
                if self._cancelled:
                    raise InterruptedError("Export cancelled by user.")
                self.progressChanged.emit(value)

            export_graded_video(
                self._source,
                self._output,
                self._adjustments,
                self._render_fn,
                codec=self._codec,
                quality=self._quality,
                copy_audio=self._copy_audio,
                progress_callback=_progress,
            )

            if self._cancelled:
                self.exportFailed.emit("Export cancelled.")
            else:
                self.exportFinished.emit(str(self._output))
        except InterruptedError:
            self.exportFailed.emit("Export cancelled.")
        except Exception as exc:
            _LOGGER.exception("Video export failed")
            self.exportFailed.emit(str(exc))
