"""Frame and contact-sheet extraction helpers.

Key optimisation: ``_build_contact_sheet_cmd`` uses ffmpeg's ``tile`` filter
to generate a *single* horizontal strip image in one pass.  The caller
receives one BGRA buffer instead of N individual frames, eliminating:
  - N signal/slot round-trips
  - N QImage / QPixmap constructions
  - N ``scaledToHeight`` calls
  - N repaints
"""

from __future__ import annotations

import bisect
import concurrent.futures
import os
import subprocess
import sys
import time

from config import PYAV_MAX_WORKERS, MAX_FFMPEG_SLICES, FRAME_READ_BUFFER
from hwaccel import _detect_hwaccel, _build_hwaccel_output_format

try:
    import av as _av_module
    HAS_PYAV = True
except ImportError:
    _av_module = None
    HAS_PYAV = False

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None


# ---------------------------------------------------------------------------
# OS helpers
# ---------------------------------------------------------------------------

def _build_popen_priority_kwargs():
    """Build OS-specific kwargs to lower the priority of ffmpeg child processes."""
    popen_kwargs = {}
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        # BELOW_NORMAL_PRIORITY_CLASS on Windows
        popen_kwargs['creationflags'] = 0x00004000
    else:
        popen_kwargs['preexec_fn'] = lambda: os.nice(10)
    return startupinfo, popen_kwargs


# =====================================================================
# Contact-sheet / strip extraction  (NEW — primary strategy)
# =====================================================================

def _build_contact_sheet_cmd(video_path, thumb_w, thumb_h, count,
                             duration, *, use_hwaccel=True,
                             keyframe_only=True):
    """Build an ffmpeg command that produces a **single** tiled strip image.

    The output is one raw BGRA frame of size ``(count * thumb_w) × thumb_h``.
    Using the ``tile`` filter avoids per-frame Python ↔ Qt overhead entirely.

    Parameters
    ----------
    video_path : str
    thumb_w, thumb_h : int
        Dimensions of each individual thumbnail cell.
    count : int
        Number of thumbnails to tile horizontally.
    duration : float
        Video duration in seconds (used to compute fps rate).
    use_hwaccel : bool
        Whether to try the detected hardware accelerator.
    keyframe_only : bool
        Whether to decode only keyframes (``-skip_frame nokey``).

    Returns
    -------
    tuple[list[str], int, int]
        (ffmpeg_cmd, strip_width, strip_height)
    """
    fps_rate = count / max(duration, 0.01)
    strip_w = count * thumb_w
    strip_h = thumb_h

    hw = _detect_hwaccel()
    hwaccel = hw['hwaccel'] if use_hwaccel else None

    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
           '-probesize', '32768', '-analyzeduration', '0',
           '-fflags', '+nobuffer']

    # Determine whether we can use the full GPU pipeline:
    #   GPU scale → hwdownload → format=bgra → (CPU fps, tile)
    # This requires -hwaccel_output_format to keep frames in GPU memory
    # for GPU scaling, then hwdownload transfers them to CPU *after* scale.
    # CPU-only filters (fps, tile) come after the download.
    gpu_scale = False
    if hwaccel:
        scale_filter = hw['scale_filter']
        download = hw['download_filter']
        if scale_filter.startswith('scale_') and download:
            hw_out_fmt = _build_hwaccel_output_format(hwaccel)
            cmd.extend(['-hwaccel', hwaccel,
                        '-hwaccel_output_format', hw_out_fmt])
            gpu_scale = True
        else:
            # HW decode only — frames auto-transfer to CPU (no GPU scale)
            cmd.extend(['-hwaccel', hwaccel])

    if keyframe_only:
        cmd.extend(['-skip_frame', 'nokey'])

    cmd.extend(['-i', video_path])

    # Build the filter graph.
    # Note: -skip_frame nokey already limits the decoder to keyframes,
    # so the redundant select='eq(pict_type,I)' filter is not needed.
    parts = []

    if gpu_scale:
        # GPU scale first (while frames are still in GPU memory),
        # then download to CPU, then CPU-only filters.
        # format=nv12 pins the hwdownload output format to prevent
        # downstream negotiation requesting unsupported formats.
        parts.append(f'{scale_filter}={thumb_w}:{thumb_h}')
        parts.append(download)
        parts.append('format=nv12')
        parts.append('format=bgra')
        parts.append(f'fps={fps_rate:.6f}')
    else:
        parts.append(f'fps={fps_rate:.6f}')
        parts.append(f'scale={thumb_w}:{thumb_h}')
        parts.append('format=bgra')

    # tile=Nx1 assembles N frames into a single horizontal row.
    # padding=0 removes any gap between cells.
    parts.append(f'tile={count}x1:padding=0')

    vf = ','.join(parts)
    cmd.extend(['-vf', vf, '-an', '-frames:v', '1',
                '-f', 'rawvideo', '-vsync', 'vfr', 'pipe:1'])

    return cmd, strip_w, strip_h


def _run_contact_sheet(video_path, thumb_w, thumb_h, count, duration,
                       *, use_hwaccel=True, keyframe_only=True):
    """Run the contact-sheet command and return the raw strip buffer.

    Returns
    -------
    tuple[int, int, bytes] | None
        (strip_width, strip_height, bgra_bytes) on success, else None.
    """
    cmd, strip_w, strip_h = _build_contact_sheet_cmd(
        video_path, thumb_w, thumb_h, count, duration,
        use_hwaccel=use_hwaccel, keyframe_only=keyframe_only,
    )
    expected_size = strip_w * strip_h * 4

    try:
        startupinfo, popen_kwargs = _build_popen_priority_kwargs()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=expected_size,
            startupinfo=startupinfo,
            **popen_kwargs,
        )

        data = proc.stdout.read(expected_size + 4096)
        proc.stdout.close()

        try:
            stderr = proc.stderr.read()
            proc.stderr.close()
        except Exception:
            stderr = b''

        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        if len(data) >= expected_size:
            return (strip_w, strip_h, bytes(data[:expected_size]))

        if stderr:
            msg = stderr[:300].decode('utf-8', errors='replace')
            print(f"[contact-sheet] Failed: {msg.strip()}")
        elif len(data) > 0:
            print(f"[contact-sheet] Short read: got {len(data)}, "
                  f"expected {expected_size}")
        return None

    except Exception as e:
        print(f"[contact-sheet] Error: {e}")
        return None


# =====================================================================
# Single-pass raw-frame extraction  (Strategy 2 — kept as fallback)
# =====================================================================

def _build_single_pass_cmd(video_path, thumb_w, thumb_h, fps_rate,
                           hwaccel=True, keyframe_only=True):
    """
    Build a single ffmpeg command that extracts ALL timeline thumbnails
    in one pass, outputting a continuous rawvideo BGRA stream to stdout.

    Uses the detected hwaccel when *hwaccel=True* (no ``-hwaccel auto``).
    """
    hw = _detect_hwaccel()
    hwaccel_name = hw['hwaccel'] if hwaccel else None

    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
           '-nostdin',
           '-probesize', '32768', '-analyzeduration', '0',
           '-fflags', '+nobuffer']

    # Determine whether we can use the full GPU pipeline.
    gpu_scale = False
    if hwaccel_name:
        scale_filter = hw['scale_filter']
        download = hw['download_filter']
        if scale_filter.startswith('scale_') and download:
            hw_out_fmt = _build_hwaccel_output_format(hwaccel_name)
            cmd.extend(['-hwaccel', hwaccel_name,
                        '-hwaccel_output_format', hw_out_fmt])
            gpu_scale = True
        else:
            cmd.extend(['-hwaccel', hwaccel_name])

    if keyframe_only:
        cmd.extend(['-skip_frame', 'nokey'])

    cmd.extend(['-i', video_path])

    # Build filter chain.
    # -skip_frame nokey already limits the decoder to keyframes,
    # so select='eq(pict_type,I)' is not needed.
    parts = []

    if gpu_scale:
        # format=nv12 pins hwdownload output to prevent negotiation errors
        parts.append(f'{scale_filter}={thumb_w}:{thumb_h}')
        parts.append(download)
        parts.append('format=nv12')
        parts.append('format=bgra')
        parts.append(f'fps={fps_rate:.6f}')
    else:
        parts.append(f'fps={fps_rate:.6f}')
        parts.append(f'scale={thumb_w}:{thumb_h}')
        parts.append('format=bgra')

    vf = ','.join(parts)
    cmd.extend(['-vf', vf, '-an', '-f', 'rawvideo', '-vsync', 'vfr',
                'pipe:1'])
    return cmd


# =====================================================================
# Pipe-based single-frame extraction (GPU → auto → SW fallback)
# =====================================================================

def _extract_frame_pipe(video_path, timestamp, thumb_w, thumb_h):
    """
    Extract a single frame as raw BGRA pixels via pipe.

    Fallback order:
      1. Specific GPU decode + GPU scale (d3d11va/cuda/videotoolbox/vaapi)
      2. Software decode + CPU scale

    Returns (width, height, bytes) on success, or None on failure.
    """
    hw = _detect_hwaccel()

    # --- Attempt 1: Specific GPU decode + GPU/CPU scale ---
    if hw['hwaccel'] is not None:
        result = _try_extract_pipe_hwaccel(
            video_path, timestamp, thumb_w, thumb_h, hw,
        )
        if result is not None:
            return result

    # --- Attempt 2: Software decode + pipe ---
    result = _try_extract_pipe_sw(video_path, timestamp, thumb_w, thumb_h)
    if result is not None:
        return result

    return None


def _try_extract_pipe_hwaccel(video_path, timestamp, thumb_w, thumb_h, hw):
    """GPU-accelerated single-frame extraction via rawvideo pipe."""
    hwaccel = hw['hwaccel']
    hw_out_fmt = _build_hwaccel_output_format(hwaccel)
    scale_filter = hw['scale_filter']
    download = hw['download_filter']

    # Build the -vf filter chain
    if scale_filter.startswith('scale_') and download:
        vf = f"{scale_filter}={thumb_w}:{thumb_h},{download},format=bgra"
    elif download:
        vf = f"{download},scale={thumb_w}:{thumb_h},format=bgra"
    else:
        vf = f"scale={thumb_w}:{thumb_h},format=bgra"

    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-nostdin',
        '-probesize', '32768', '-analyzeduration', '0',
        '-fflags', '+nobuffer',
        '-hwaccel', hwaccel,
        '-hwaccel_output_format', hw_out_fmt,
        '-ss', f'{timestamp:.4f}',
        '-i', video_path,
        '-frames:v', '1',
        '-vf', vf,
        '-f', 'rawvideo',
        'pipe:1',
    ]

    return _run_pipe_cmd(cmd, thumb_w, thumb_h)


def _try_extract_pipe_sw(video_path, timestamp, thumb_w, thumb_h):
    """Software-only single-frame extraction via rawvideo pipe."""
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-nostdin',
        '-probesize', '32768', '-analyzeduration', '0',
        '-fflags', '+nobuffer',
        '-ss', f'{timestamp:.4f}',
        '-i', video_path,
        '-frames:v', '1',
        '-vf', f'scale={thumb_w}:{thumb_h},format=bgra',
        '-f', 'rawvideo',
        'pipe:1',
    ]

    return _run_pipe_cmd(cmd, thumb_w, thumb_h)


def _try_extract_pipe_auto(video_path, timestamp, thumb_w, thumb_h):
    """GPU auto-detect decode + CPU scale via rawvideo pipe.

    Uses '-hwaccel auto' which lets ffmpeg pick the best available hardware
    decoder.  Kept as a compatibility path for edge cases.
    """
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-nostdin',
        '-probesize', '32768', '-analyzeduration', '0',
        '-fflags', '+nobuffer',
        '-hwaccel', 'auto',
        '-ss', f'{timestamp:.4f}',
        '-i', video_path,
        '-frames:v', '1',
        '-vf', f'scale={thumb_w}:{thumb_h},format=bgra',
        '-f', 'rawvideo',
        'pipe:1',
    ]

    return _run_pipe_cmd(cmd, thumb_w, thumb_h)


def _run_pipe_cmd(cmd, expected_w, expected_h):
    """Run an ffmpeg command that outputs rawvideo BGRA to stdout pipe.

    Returns (width, height, bytes) or None on failure.
    """
    expected_size = expected_w * expected_h * 4

    try:
        startupinfo, popen_kwargs = _build_popen_priority_kwargs()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            **popen_kwargs,
        )

        if proc.returncode != 0:
            stderr_msg = proc.stderr[:300] if proc.stderr else b''
            if isinstance(stderr_msg, bytes):
                stderr_msg = stderr_msg.decode('utf-8', errors='replace')
            hwaccel_in_cmd = any(x in cmd for x in ['-hwaccel'])
            label = "GPU" if hwaccel_in_cmd else "SW"
            print(f"[ffmpeg {label}] exit={proc.returncode}: {stderr_msg.strip()}")
            return None

        if len(proc.stdout) != expected_size:
            print(f"[ffmpeg] Unexpected frame size: got {len(proc.stdout)}, "
                  f"expected {expected_size} ({expected_w}x{expected_h}x4)")
            return None

        return (expected_w, expected_h, proc.stdout)
    except Exception as e:
        print(f"[ffmpeg] Pipe extraction error: {e}")
        return None


def _extract_single_frame(args):
    """Extract exactly one frame at a specific timestamp.

    Returns either:
      - ('pipe', width, height, bytes)  for pipe-based extraction, or
      - ('file', path)                  for file-based fallback, or
      - None                            on total failure.
    """
    video_path = args[0]
    timestamp = args[1]
    target_height = args[2]
    out_path = args[3]

    # Extended args format: (video_path, timestamp, target_height, out_path, thumb_w)
    if len(args) == 5:
        thumb_w = args[4]
    else:
        thumb_w = None

    if thumb_w is not None and thumb_w > 0:
        result = _extract_frame_pipe(video_path, timestamp, thumb_w, target_height)
        if result is not None:
            w, h, buf = result
            return ('pipe', w, h, buf)

    # --- Fallback: file-based extraction (original approach) ---
    cmd = [
        'ffmpeg', '-nostdin',
        '-probesize', '32768', '-analyzeduration', '0',
        '-fflags', '+nobuffer',
        '-ss', f'{timestamp:.4f}',
        '-i', video_path,
        '-vf', f'scale=-1:{target_height}',
        '-frames:v', '1',
        '-q:v', '3',
        '-y',
        out_path,
    ]

    try:
        startupinfo, popen_kwargs = _build_popen_priority_kwargs()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
            **popen_kwargs,
        )
        proc.wait()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return ('file', out_path)
    except Exception as e:
        print(f"FFmpeg frame extraction error: {e}")
    return None


# =====================================================================
# PyAV-based extraction
# =====================================================================

_HW_PIX_FMTS = frozenset([
    'cuda', 'dxva2_vld', 'd3d11', 'vaapi', 'vdpau',
    'videotoolbox_vld', 'qsv',
])


def _get_keyframe_timestamps_pyav(video_path):
    """Extract all keyframe timestamps using PyAV packet-level demux.

    Returns a sorted list of float timestamps in seconds.
    """
    keyframes = []
    container = None
    try:
        container = _av_module.open(video_path)
        stream = container.streams.video[0]
        time_base = stream.time_base
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            if packet.is_keyframe:
                t = float(packet.pts * time_base)
                keyframes.append(t)
    except Exception as e:
        print(f"[pyav-keyframes] Error: {e}")
    finally:
        if container:
            try:
                container.close()
            except Exception:
                pass
    if not keyframes:
        return keyframes
    return sorted(set(keyframes))


def _snap_to_keyframes(target_times, keyframes):
    """Map each target time to the nearest keyframe timestamp.

    Returns list of (original_index, snapped_timestamp) pairs.
    """
    if not keyframes:
        return list(enumerate(target_times))

    snapped = []
    for i, t in enumerate(target_times):
        pos = bisect.bisect_left(keyframes, t)
        candidates = []
        if pos < len(keyframes):
            candidates.append(keyframes[pos])
        if pos > 0:
            candidates.append(keyframes[pos - 1])
        best = min(candidates, key=lambda k: abs(k - t))
        snapped.append((i, best))
    return snapped


def _pyav_extract_segment(video_path, indices, thumb_w, thumb_h,
                          rotation=0, vflip=False):
    """Extract a subset of frames using individual seeks.

    Returns list of (global_index, width, height, rgb_bytes) tuples.
    """
    if not indices:
        return []

    results = []
    container = None
    try:
        container = _av_module.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 2
        time_base = stream.time_base

        # PyAV gives raw (unrotated) frames → extract at raw dimensions
        if rotation in (90, 270):
            raw_w, raw_h = thumb_h, thumb_w
        else:
            raw_w, raw_h = thumb_w, thumb_h

        for global_idx, target_time in indices:
            target_pts = int(target_time / float(time_base))
            container.seek(max(0, target_pts), stream=stream)

            for frame in container.decode(stream):
                img = frame.to_image(
                    width=raw_w, height=raw_h,
                    interpolation='FAST_BILINEAR',
                )

                # Apply orientation transforms (PyAV does NOT auto-rotate)
                if rotation == 90:
                    img = img.transpose(_PILImage.Transpose.ROTATE_270)
                elif rotation == 180:
                    img = img.transpose(_PILImage.Transpose.ROTATE_180)
                elif rotation == 270:
                    img = img.transpose(_PILImage.Transpose.ROTATE_90)
                if vflip:
                    img = img.transpose(_PILImage.Transpose.FLIP_TOP_BOTTOM)

                rgb_data = img.tobytes("raw", "RGB")
                results.append((
                    global_idx, thumb_w, thumb_h, rgb_data,
                ))
                break  # Only need first frame after seek

    except Exception as e:
        print(f"[pyav-segment] Error: {e}")
    finally:
        if container:
            try:
                container.close()
            except Exception:
                pass
    return results


def _extract_thumbnails_pyav(video_path, num_frames, thumb_w, thumb_h,
                             callback=None, rotation=0, vflip=False):
    """Extract thumbnails using optimised keyframe-aware PyAV.

    Pipeline:
    1. Single container open for duration probe **and** keyframe scan —
       saves one container open vs. the previous two-open approach.
    2. Snap target times to nearest keyframes, then **deduplicate** so
       each keyframe is decoded at most once.
    3. Single container with sequential forward seeks and
       ``thread_count=0`` (all CPU cores for the decoder).  Forward-only
       seeks are nearly free since the demuxer doesn't need to re-scan.

    Returns list of (width, height, bytes) tuples in RGB888 format.
    """
    try:
        # --- Phase 1: probe duration + scan keyframes in ONE open ---
        container = _av_module.open(video_path)
        stream = container.streams.video[0]
        time_base = stream.time_base

        duration = 0.0
        if stream.duration and time_base:
            duration = float(stream.duration * time_base)
        if duration <= 0 and container.duration:
            duration = container.duration / _av_module.time_base

        keyframes = []
        if duration > 0:
            for packet in container.demux(stream):
                if packet.pts is None:
                    continue
                if packet.is_keyframe:
                    keyframes.append(float(packet.pts * time_base))
        container.close()

        if duration <= 0:
            return []

        if keyframes:
            keyframes = sorted(set(keyframes))

        # --- Phase 2: compute targets, snap, deduplicate ---
        step = duration / num_frames
        target_times = [i * step for i in range(num_frames)]

        if keyframes:
            all_indices = _snap_to_keyframes(target_times, keyframes)
            print(f"[pyav] Snapped {num_frames} targets to "
                  f"{len(keyframes)} keyframes")
        else:
            all_indices = list(enumerate(target_times))

        # Deduplicate: group target indices by their snapped keyframe time
        # so each unique keyframe is decoded only once.
        kf_to_indices: dict[float, list[int]] = {}
        for global_idx, kf_time in all_indices:
            kf_key = round(kf_time, 6)
            kf_to_indices.setdefault(kf_key, []).append(global_idx)

        unique_times = sorted(kf_to_indices.keys())
        unique_count = len(unique_times)
        print(f"[pyav] {unique_count} unique keyframes to decode")

        # --- Phase 3: single-container sequential forward seeks ---
        if rotation in (90, 270):
            raw_w, raw_h = thumb_h, thumb_w
        else:
            raw_w, raw_h = thumb_w, thumb_h

        container = _av_module.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        # Use all CPU cores for the single decoder — faster than
        # splitting across multiple containers with limited threads.
        stream.codec_context.thread_count = 0
        time_base = stream.time_base

        frame_cache: dict[float, tuple[int, int, bytes]] = {}

        for kf_time in unique_times:
            target_pts = int(kf_time / float(time_base))
            container.seek(max(0, target_pts), stream=stream)

            for frame in container.decode(stream):
                img = frame.to_image(
                    width=raw_w, height=raw_h,
                    interpolation='FAST_BILINEAR',
                )

                # Apply orientation transforms (PyAV does NOT auto-rotate)
                if rotation == 90:
                    img = img.transpose(_PILImage.Transpose.ROTATE_270)
                elif rotation == 180:
                    img = img.transpose(_PILImage.Transpose.ROTATE_180)
                elif rotation == 270:
                    img = img.transpose(_PILImage.Transpose.ROTATE_90)
                if vflip:
                    img = img.transpose(_PILImage.Transpose.FLIP_TOP_BOTTOM)

                rgb_data = img.tobytes("raw", "RGB")
                kf_key = round(kf_time, 6)
                frame_cache[kf_key] = (thumb_w, thumb_h, rgb_data)
                break  # Only need first frame after seek

        container.close()

        # --- Phase 4: emit results in target order ---
        thumbnails = []
        for global_idx, kf_time in sorted(all_indices, key=lambda x: x[0]):
            kf_key = round(kf_time, 6)
            if kf_key in frame_cache:
                w, h, rgb_data = frame_cache[kf_key]
                thumbnails.append((w, h, rgb_data))
                if callback:
                    callback(global_idx, rgb_data, w, h)

        return thumbnails

    except Exception as e:
        print(f"[pyav] Extraction error: {e}")
        return []
