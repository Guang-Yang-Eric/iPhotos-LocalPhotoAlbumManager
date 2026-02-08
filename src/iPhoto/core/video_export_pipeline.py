# -*- coding: utf-8 -*-
"""Core abstraction for the video export rendering pipeline.

This module is independent of the GUI layer and provides a pure-logic interface
for decoding a source video, applying adjustments frame-by-frame via an
offscreen GL renderer, and writing the result to an output file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

_LOGGER = logging.getLogger(__name__)


def iterate_decoded_frames(
    source: Path,
    *,
    target_size: Optional[tuple[int, int]] = None,
) -> Iterator[Any]:
    """Yield decoded video frames from *source* as PIL ``Image`` objects.

    Each yielded value is a ``PIL.Image.Image`` in RGB mode.  When
    *target_size* is given the frames are scaled to fit within
    ``(max_width, max_height)`` while preserving aspect ratio.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for video frame iteration.") from exc

    with av.open(str(source)) as container:
        if not container.streams.video:
            return
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for frame in container.decode(stream):
            image = frame.to_image()

            if target_size is not None:
                max_w, max_h = target_size
                w, h = image.size
                ratio = min(max_w / w, max_h / h) if w and h else 1.0
                if ratio < 1.0:
                    new_w = max(2, int((w * ratio) / 2) * 2)
                    new_h = max(2, int((h * ratio) / 2) * 2)
                    image = image.resize((new_w, new_h))

            yield image


def get_video_metadata(source: Path) -> dict[str, Any]:
    """Return basic video metadata (fps, duration, resolution).

    Falls back to sensible defaults when fields are missing.
    """
    try:
        import av
    except ImportError:
        return {"fps": 30.0, "duration": 0.0, "width": 0, "height": 0}

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                return {"fps": 30.0, "duration": 0.0, "width": 0, "height": 0}
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
            return {
                "fps": fps,
                "duration": duration,
                "width": stream.width,
                "height": stream.height,
            }
    except Exception:
        return {"fps": 30.0, "duration": 0.0, "width": 0, "height": 0}


def export_graded_video(
    source: Path,
    output: Path,
    adjustments: Mapping[str, object],
    render_fn: Callable,
    *,
    codec: str = "libx264",
    quality: int = 23,
    copy_audio: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> None:
    """Render *source* with colour grading and write to *output*.

    Parameters
    ----------
    source:
        Original video file.
    output:
        Destination path.
    adjustments:
        The ``EditSession`` adjustments dict.
    render_fn:
        ``(pil_image, adjustments) -> QImage`` callable that performs offscreen
        GL rendering for a single frame.
    codec:
        FFmpeg video codec.
    quality:
        CRF value.
    copy_audio:
        Whether to copy the original audio stream.
    progress_callback:
        Optional ``(0.0 … 1.0)`` progress callback.
    """
    from ..utils.ffmpeg import encode_video_from_frames

    meta = get_video_metadata(source)
    fps = meta["fps"]
    total_frames = int(fps * meta["duration"]) if meta["duration"] > 0 else 0

    def _frame_gen():
        for idx, pil_frame in enumerate(iterate_decoded_frames(source)):
            graded = render_fn(pil_frame, adjustments)
            if progress_callback and total_frames > 0:
                progress_callback((idx + 1) / total_frames)
            yield graded

    encode_video_from_frames(
        output,
        _frame_gen(),
        fps=fps,
        audio_source=source if copy_audio else None,
        codec=codec,
        quality=quality,
    )
