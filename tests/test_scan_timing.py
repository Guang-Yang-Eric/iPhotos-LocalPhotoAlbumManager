"""Tests for GUI scan timing: elapsed time stored in StatusBarController and
the scanElapsed signal emitted by LibraryUpdateService."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication([])


def _make_status_bar_controller():
    """Return a :class:`StatusBarController` wired to lightweight Qt widgets."""
    _ensure_qapp()

    from PySide6.QtWidgets import QProgressBar, QWidget
    from iPhoto.gui.ui.widgets.chrome_status_bar import ChromeStatusBar
    from iPhoto.gui.ui.controllers.status_bar_controller import StatusBarController

    # Use real (but invisible) widgets so QObject parent chaining is satisfied.
    status_bar = ChromeStatusBar()
    progress_bar = QProgressBar()
    context = MagicMock()

    ctrl = StatusBarController(
        status_bar=status_bar,
        progress_bar=progress_bar,
        rescan_action=None,
        context=context,
    )
    # Patch show_message so we can inspect calls without needing a visible window.
    ctrl._shown_messages: list[str] = []
    original_show = ctrl.show_message

    def _capturing_show(message: str, timeout_ms=None) -> None:
        ctrl._shown_messages.append(message)
        original_show(message, timeout_ms)

    ctrl.show_message = _capturing_show  # type: ignore[method-assign]
    return ctrl


# ---------------------------------------------------------------------------
# StatusBarController – elapsed-time tests
# ---------------------------------------------------------------------------


class TestStatusBarControllerScanElapsed:
    """StatusBarController tracks and displays scan elapsed time."""

    def test_handle_scan_elapsed_stores_value(self):
        """handle_scan_elapsed must persist the elapsed seconds for later use."""
        ctrl = _make_status_bar_controller()
        ctrl.handle_scan_elapsed(Path("/photos"), 3.14)
        assert ctrl._last_scan_elapsed_s == pytest.approx(3.14)

    def test_handle_scan_elapsed_overwrites_previous_value(self):
        """A new scanElapsed must replace the stale previous value."""
        ctrl = _make_status_bar_controller()
        ctrl.handle_scan_elapsed(Path("/a"), 1.0)
        ctrl.handle_scan_elapsed(Path("/b"), 5.5)
        assert ctrl._last_scan_elapsed_s == pytest.approx(5.5)

    def test_handle_scan_finished_includes_elapsed_when_available(self):
        """handle_scan_finished must show 'Scan complete. (X.Xs)' when elapsed is known."""
        ctrl = _make_status_bar_controller()
        ctrl.handle_scan_elapsed(Path("/photos"), 2.3)
        ctrl.handle_scan_finished(Path("/photos"), success=True)

        assert ctrl._shown_messages, "show_message should have been called"
        last_msg = ctrl._shown_messages[-1]
        assert "Scan complete." in last_msg
        assert "2.3" in last_msg

    def test_handle_scan_finished_plain_message_without_elapsed(self):
        """handle_scan_finished must show 'Scan complete.' when no elapsed is known."""
        ctrl = _make_status_bar_controller()
        ctrl.handle_scan_finished(Path("/photos"), success=True)

        assert ctrl._shown_messages, "show_message should have been called"
        last_msg = ctrl._shown_messages[-1]
        assert last_msg == "Scan complete."

    def test_handle_scan_finished_failure_message_unchanged(self):
        """Scan failure message must not include elapsed time."""
        ctrl = _make_status_bar_controller()
        ctrl.handle_scan_elapsed(Path("/photos"), 1.0)
        ctrl.handle_scan_finished(Path("/photos"), success=False)

        assert ctrl._shown_messages
        last_msg = ctrl._shown_messages[-1]
        assert last_msg == "Scan failed."

    def test_handle_scan_finished_clears_elapsed(self):
        """After handle_scan_finished the stored elapsed must be reset to None."""
        ctrl = _make_status_bar_controller()
        ctrl.handle_scan_elapsed(Path("/photos"), 1.0)
        ctrl.handle_scan_finished(Path("/photos"), success=True)
        assert ctrl._last_scan_elapsed_s is None

    def test_second_scan_finished_without_elapsed_shows_plain_message(self):
        """When the facade emits an extra scanFinished with no paired scanElapsed
        (e.g. from the library manager relay), the message must be plain."""
        ctrl = _make_status_bar_controller()
        # First scan – full path with timing.
        ctrl.handle_scan_elapsed(Path("/photos"), 0.5)
        ctrl.handle_scan_finished(Path("/photos"), success=True)
        # Second scanFinished arrives from the facade relay (no paired scanElapsed).
        ctrl.handle_scan_finished(Path("/photos"), success=True)

        last_msg = ctrl._shown_messages[-1]
        assert last_msg == "Scan complete."


# ---------------------------------------------------------------------------
# LibraryUpdateService._emit_scan_elapsed – unit tests
# ---------------------------------------------------------------------------


class TestLibraryUpdateServiceScanElapsed:
    """LibraryUpdateService._emit_scan_elapsed emits the signal and resets state."""

    def _make_service(self):
        """Build a minimal LibraryUpdateService with all heavy deps mocked."""
        _ensure_qapp()

        from iPhoto.gui.services.library_update_service import LibraryUpdateService
        from iPhoto.gui.background_task_manager import BackgroundTaskManager

        task_manager = MagicMock(spec=BackgroundTaskManager)
        svc = LibraryUpdateService(
            task_manager=task_manager,
            current_album_getter=lambda: None,
            library_manager_getter=lambda: None,
        )
        return svc

    def test_emit_scan_elapsed_resets_start_time(self):
        """_emit_scan_elapsed must clear _scan_start_time after emission."""
        svc = self._make_service()
        svc._scan_start_time = time.perf_counter()
        svc._emit_scan_elapsed(Path("/photos"))
        assert svc._scan_start_time is None

    def test_emit_scan_elapsed_defaults_to_zero_when_no_start_time(self):
        """When _scan_start_time was never set, elapsed should default to 0.0."""
        from PySide6.QtTest import QSignalSpy

        svc = self._make_service()
        assert svc._scan_start_time is None

        spy = QSignalSpy(svc.scanElapsed)
        svc._emit_scan_elapsed(Path("/photos"))

        assert spy.count() == 1
        args = spy.at(0)
        elapsed = args[1]
        assert float(elapsed) == pytest.approx(0.0)

    def test_emit_scan_elapsed_signal_carries_non_negative_elapsed(self):
        """scanElapsed must carry a non-negative float after a real start time."""
        from PySide6.QtTest import QSignalSpy

        svc = self._make_service()
        svc._scan_start_time = time.perf_counter()

        spy = QSignalSpy(svc.scanElapsed)
        svc._emit_scan_elapsed(Path("/photos"))

        assert spy.count() == 1
        args = spy.at(0)
        elapsed = float(args[1])
        assert elapsed >= 0.0

    def test_emit_scan_elapsed_signal_carries_correct_root(self):
        """The Path emitted by scanElapsed must match the argument to _emit_scan_elapsed."""
        from PySide6.QtTest import QSignalSpy

        target = Path("/my/album")
        svc = self._make_service()
        spy = QSignalSpy(svc.scanElapsed)
        svc._emit_scan_elapsed(target)

        assert spy.count() == 1
        args = spy.at(0)
        emitted_root = args[0]
        assert Path(emitted_root) == target

    def test_rescan_album_async_sets_scan_start_time(self):
        """rescan_album_async must record _scan_start_time before submitting the task."""
        svc = self._make_service()

        album = MagicMock()
        album.root = Path("/photos")
        album.manifest = {}

        # Intercept submit_task so the worker never actually runs.
        svc._task_manager.submit_task = MagicMock()

        assert svc._scan_start_time is None
        svc.rescan_album_async(album)
        assert svc._scan_start_time is not None
