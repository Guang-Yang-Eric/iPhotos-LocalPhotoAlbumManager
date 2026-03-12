"""ThumbnailWorker — QThread that generates timeline thumbnails.

Strategy order (optimised for "one-shot complete display"):
  0. PyAV in-process extraction — zero subprocess overhead, keyframe-aware
  1. Contact-sheet strip (GPU + keyframe) — single image, zero per-frame cost
  2. Contact-sheet strip (CPU + keyframe)
  3. Contact-sheet strip (CPU, full decode)
  4. Single-pass raw frames (GPU + keyframe)
  5. Single-pass raw frames (CPU + keyframe)
  6. Sliced multi-process extraction
  7. Parallel individual extraction (slowest fallback)
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import time

from PySide6.QtCore import QThread, Signal

from config import (
    THUMB_WIDTH, FRAME_READ_BUFFER, MAX_FFMPEG_SLICES, HIDPI_MODE,
    THUMB_LOGICAL_HEIGHT,
)
from probe import HAS_PYAV, _get_video_info, _get_video_info_pyav
from extraction import (
    _run_contact_sheet,
    _build_single_pass_cmd,
    _build_popen_priority_kwargs,
    _extract_single_frame,
    _extract_thumbnails_pyav,
)


class ThumbnailWorker(QThread):
    """Background thread that generates timeline thumbnails.

    Prefers the contact-sheet strategy which produces a single tiled
    strip image in one ffmpeg call — the UI receives one QPixmap and
    paints it once.  Falls back through progressively slower strategies
    if needed.
    """
    # Emits a single strip image: ('strip', width, height, bgra_bytes)
    # or a single thumbnail:      ('pipe'|'pyav', width, height, bytes)
    thumbnail_ready = Signal(object)
    # Batch fallback: emits all results at once
    thumbnails_ready = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, video_path, target_height, visible_width, temp_dir,
                 num_workers=None, dpr=1.0, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.target_height = target_height
        self.visible_width = visible_width
        self.temp_dir = temp_dir
        self.dpr = dpr
        self._abort = False
        self._proc = None
        if num_workers is None:
            num_workers = os.cpu_count() or 4
        self.num_workers = num_workers

    def abort(self):
        """Request the worker to stop. Kills any running ffmpeg process."""
        self._abort = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.kill()
            except OSError:
                pass

    def run(self):
        try:
            t0 = time.perf_counter()

            # --- Probe video info (prefer PyAV, fallback to ffprobe) ---
            rotation = 0
            vflip = False
            if HAS_PYAV:
                v_w, v_h, duration, rotation, vflip = \
                    _get_video_info_pyav(self.video_path)
            else:
                v_w, v_h, duration = 0, 0, 0
            if v_w <= 0 or v_h <= 0 or duration <= 0:
                v_w, v_h, duration, rotation, vflip = \
                    _get_video_info(self.video_path)
            if v_w <= 0 or v_h <= 0 or duration <= 0:
                self.error_occurred.emit("Failed to probe video")
                return

            # Pre-compute thumbnail dimensions from video aspect ratio
            thumb_w = int(v_w * (self.target_height / v_h))
            thumb_w = max(2, thumb_w + (thumb_w % 2))
            target_h = max(2, self.target_height + (self.target_height % 2))

            scaled_width = v_w * (self.target_height / v_h)
            if scaled_width <= 0:
                scaled_width = THUMB_WIDTH
            # scaled_width is in physical pixels; convert to logical for
            # count_needed (visible_width is in logical pixels)
            logical_thumb_w = scaled_width / max(self.dpr, 1.0)
            count_needed = int(self.visible_width / logical_thumb_w) + 2
            count_needed = max(count_needed, 5)
            count_needed = min(count_needed, 60)

            print(f"Video: {v_w}x{v_h}, Duration: {duration:.1f}s, "
                  f"Thumbnails: {count_needed} @ {thumb_w}x{target_h}, "
                  f"rotation={rotation}, vflip={vflip}, "
                  f"PyAV={'yes' if HAS_PYAV else 'no'}")

            # --- Strategy 0: PyAV in-process extraction (fastest) ---
            if HAS_PYAV and not self._abort:
                if self._try_pyav(thumb_w, target_h, count_needed,
                                  rotation, vflip):
                    elapsed = time.perf_counter() - t0
                    print(f"[thumbnail] Done in {elapsed:.2f}s (PyAV)")
                    return

            # --- Strategy 1: Contact-sheet with GPU + keyframe ---
            if not self._abort:
                result = _run_contact_sheet(
                    self.video_path, thumb_w, target_h, count_needed,
                    duration, use_hwaccel=True, keyframe_only=True,
                )
                if result is not None:
                    self.thumbnail_ready.emit(
                        ('strip', result[0], result[1], result[2]),
                    )
                    elapsed = time.perf_counter() - t0
                    print(f"[thumbnail] Done in {elapsed:.2f}s "
                          f"(contact-sheet gpu+keyframe)")
                    return

            # --- Strategy 2: Contact-sheet CPU + keyframe ---
            if not self._abort:
                result = _run_contact_sheet(
                    self.video_path, thumb_w, target_h, count_needed,
                    duration, use_hwaccel=False, keyframe_only=True,
                )
                if result is not None:
                    self.thumbnail_ready.emit(
                        ('strip', result[0], result[1], result[2]),
                    )
                    elapsed = time.perf_counter() - t0
                    print(f"[thumbnail] Done in {elapsed:.2f}s "
                          f"(contact-sheet cpu+keyframe)")
                    return

            # --- Strategy 3: Contact-sheet CPU full decode ---
            if not self._abort:
                result = _run_contact_sheet(
                    self.video_path, thumb_w, target_h, count_needed,
                    duration, use_hwaccel=False, keyframe_only=False,
                )
                if result is not None:
                    self.thumbnail_ready.emit(
                        ('strip', result[0], result[1], result[2]),
                    )
                    elapsed = time.perf_counter() - t0
                    print(f"[thumbnail] Done in {elapsed:.2f}s "
                          f"(contact-sheet cpu+full)")
                    return

            fps_rate = count_needed / max(duration, 0.01)
            frame_size = thumb_w * target_h * 4

            # --- Strategy 4: Single-pass GPU + keyframe (per-frame) ---
            if self._try_single_pass(
                thumb_w, target_h, count_needed, fps_rate, frame_size,
                hwaccel=True, keyframe_only=True,
            ):
                elapsed = time.perf_counter() - t0
                print(f"[thumbnail] Done in {elapsed:.2f}s "
                      f"(single-pass gpu+keyframe)")
                return

            # --- Strategy 5: Single-pass CPU + keyframe ---
            if self._try_single_pass(
                thumb_w, target_h, count_needed, fps_rate, frame_size,
                hwaccel=False, keyframe_only=True,
            ):
                elapsed = time.perf_counter() - t0
                print(f"[thumbnail] Done in {elapsed:.2f}s "
                      f"(single-pass keyframe)")
                return

            # --- Strategy 6: Sliced multi-process ---
            if self._try_sliced_single_pass(
                thumb_w, target_h, count_needed, duration,
                keyframe_only=True,
            ):
                elapsed = time.perf_counter() - t0
                print(f"[thumbnail] Done in {elapsed:.2f}s "
                      f"(sliced keyframe)")
                return

            # --- Strategy 7: Parallel individual (slowest fallback) ---
            self._fallback_parallel(
                thumb_w, target_h, count_needed, duration,
            )
            elapsed = time.perf_counter() - t0
            print(f"[thumbnail] Done in {elapsed:.2f}s (parallel fallback)")
        except Exception as e:
            self.error_occurred.emit(str(e))

    # -----------------------------------------------------------------
    # Strategy helpers
    # -----------------------------------------------------------------

    def _try_pyav(self, thumb_w, thumb_h, count_needed,
                  rotation=0, vflip=False):
        """Extract thumbnails using PyAV — zero subprocess overhead."""
        def on_frame(index, rgb_data, w, h):
            if not self._abort:
                self.thumbnail_ready.emit(('pyav', w, h, rgb_data))

        results = _extract_thumbnails_pyav(
            self.video_path, count_needed, thumb_w, thumb_h,
            callback=on_frame,
            rotation=rotation, vflip=vflip,
        )
        return len(results) > 0

    def _try_single_pass(self, thumb_w, thumb_h, count_needed, fps_rate,
                         frame_size, hwaccel=True, keyframe_only=True):
        """Single ffmpeg process outputting continuous rawvideo BGRA."""
        mode = "gpu" if hwaccel else "cpu"
        if keyframe_only:
            mode += "+keyframe"
        cmd = _build_single_pass_cmd(
            self.video_path, thumb_w, thumb_h, fps_rate,
            hwaccel=hwaccel, keyframe_only=keyframe_only,
        )
        print(f"[thumbnail] Trying single-pass ({mode})")

        try:
            startupinfo, popen_kwargs = _build_popen_priority_kwargs()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_size,
                startupinfo=startupinfo,
                **popen_kwargs,
            )
            self._proc = proc

            count = 0
            max_frames = count_needed + 5
            while count < max_frames and not self._abort:
                data = proc.stdout.read(frame_size)
                if len(data) < frame_size:
                    break
                self.thumbnail_ready.emit(
                    ('pipe', thumb_w, thumb_h, bytes(data)),
                )
                count += 1

            proc.stdout.close()
            try:
                stderr = proc.stderr.read()
                proc.stderr.close()
            except Exception:
                stderr = b''
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            self._proc = None

            if count > 0:
                print(f"[thumbnail] Single-pass ({mode}): "
                      f"got {count} frames")
                return True

            if stderr:
                msg = stderr[:300].decode('utf-8', errors='replace')
                print(f"[thumbnail] Single-pass ({mode}) failed: "
                      f"{msg.strip()}")
            return False

        except Exception as e:
            print(f"[thumbnail] Single-pass ({mode}) error: {e}")
            self._proc = None
            return False

    def _fallback_parallel(self, thumb_w, target_h, count_needed,
                           duration):
        """Fall back to N parallel individual frame extractions."""
        print("[thumbnail] Falling back to parallel extraction")

        timestamps = [
            i * duration / count_needed for i in range(count_needed)
        ]
        tasks = []
        for i, ts in enumerate(timestamps):
            out_path = os.path.join(
                self.temp_dir, f"thumb_{i:04d}.jpg",
            )
            tasks.append(
                (self.video_path, ts, target_h, out_path, thumb_w),
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers,
        ) as pool:
            results = list(pool.map(_extract_single_frame, tasks))

        valid = [r for r in results if r is not None]
        self.thumbnails_ready.emit(valid)

    def _try_sliced_single_pass(self, thumb_w, thumb_h, count_needed,
                                duration, keyframe_only=True):
        """Distribute thumbnail extraction across concurrent ffmpeg slices."""
        num_slices = min(MAX_FFMPEG_SLICES, max(1, (os.cpu_count() or 2) // 2))
        num_slices = min(num_slices, count_needed)
        if num_slices <= 1:
            return False

        frames_per_slice = count_needed // num_slices
        remainder = count_needed % num_slices
        slices = []
        step = duration / count_needed
        offset = 0
        for s in range(num_slices):
            n = frames_per_slice + (1 if s < remainder else 0)
            start_time = offset * step
            seg_duration = n * step
            slices.append((start_time, seg_duration, n))
            offset += n

        frame_size = thumb_w * thumb_h * 4
        mode = "sliced"
        if keyframe_only:
            mode += "+keyframe"
        print(f"[thumbnail] Trying {mode} ({num_slices} slices)")

        def run_slice(slice_args):
            s_idx, (start_t, seg_dur, n_frames) = slice_args
            fps_rate = n_frames / max(seg_dur, 0.01)
            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
                   '-nostdin',
                   '-probesize', '32768', '-analyzeduration', '0',
                   '-fflags', '+nobuffer']
            if keyframe_only:
                cmd.extend(['-skip_frame', 'nokey'])
            cmd.extend(['-ss', f'{start_t:.4f}',
                        '-t', f'{seg_dur:.4f}',
                        '-i', self.video_path])
            vf = (f"fps={fps_rate:.6f},"
                  f"scale={thumb_w}:{thumb_h},format=bgra")
            cmd.extend(['-vf', vf, '-an', '-f', 'rawvideo', 'pipe:1'])

            try:
                startupinfo, popen_kwargs = _build_popen_priority_kwargs()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=frame_size,
                    startupinfo=startupinfo,
                    **popen_kwargs,
                )
                frames = []
                max_read = n_frames + FRAME_READ_BUFFER
                for _ in range(max_read):
                    data = proc.stdout.read(frame_size)
                    if len(data) < frame_size:
                        break
                    frames.append(bytes(data))
                proc.stdout.close()
                try:
                    proc.stderr.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return s_idx, frames
            except Exception as e:
                print(f"[sliced] Slice {s_idx} error: {e}")
                return s_idx, []

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_slices,
            ) as pool:
                futures = [
                    pool.submit(run_slice, (i, s))
                    for i, s in enumerate(slices)
                ]
                slice_results = {}
                for f in concurrent.futures.as_completed(futures):
                    if self._abort:
                        break
                    s_idx, frames = f.result()
                    slice_results[s_idx] = frames

            # Merge slices in order and emit
            total = 0
            for s_idx in range(num_slices):
                for data in slice_results.get(s_idx, []):
                    if self._abort:
                        break
                    self.thumbnail_ready.emit(
                        ('pipe', thumb_w, thumb_h, data),
                    )
                    total += 1

            if total > 0:
                print(f"[thumbnail] Sliced ({mode}): got {total} frames")
                return True
            return False

        except Exception as e:
            print(f"[thumbnail] Sliced ({mode}) error: {e}")
            return False
