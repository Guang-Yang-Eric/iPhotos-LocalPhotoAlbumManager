from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from iPhoto.infrastructure.services import thumbnail_generator as thumbnail_module


def test_generate_video_thumbnail_uses_pyav_image_without_bytes_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.touch()
    pyav_image = Image.new("RGB", (64, 36), color="purple")

    monkeypatch.setattr(
        thumbnail_module,
        "extract_frame_with_pyav",
        lambda *args, **kwargs: pyav_image,
    )
    monkeypatch.setattr(
        thumbnail_module,
        "_extract_video_frame_with_fallbacks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bytes fallback should not run when PyAV succeeds")
        ),
    )

    generator = thumbnail_module.PillowThumbnailGenerator()
    thumbnail = generator._generate_video_thumbnail(video_path, (96, 96))

    assert thumbnail is not None
    assert thumbnail.size == (64, 36)
    assert thumbnail is not pyav_image


def test_generate_video_thumbnail_falls_back_to_bytes_when_pyav_returns_none(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.touch()
    expected = Image.new("RGB", (80, 45), color="green")
    buffer = io.BytesIO()
    expected.save(buffer, format="JPEG")

    monkeypatch.setattr(
        thumbnail_module,
        "extract_frame_with_pyav",
        lambda *args, **kwargs: None,
    )

    calls: list[tuple[Path, float, tuple[int, int], str, bool]] = []

    def fake_fallback(
        source: Path,
        *,
        at: float,
        scale: tuple[int, int],
        format: str,
        allow_pyav: bool,
    ) -> bytes:
        calls.append((source, at, scale, format, allow_pyav))
        return buffer.getvalue()

    monkeypatch.setattr(
        thumbnail_module,
        "_extract_video_frame_with_fallbacks",
        fake_fallback,
    )

    generator = thumbnail_module.PillowThumbnailGenerator()
    thumbnail = generator._generate_video_thumbnail(video_path, (96, 96))

    assert thumbnail is not None
    assert thumbnail.size == (80, 45)
    assert calls == [(video_path, 0.0, (96, 96), "jpeg", False)]
