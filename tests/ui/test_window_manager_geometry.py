"""Regression tests for frameless window geometry clamping across screen changes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from PySide6.QtCore import QPoint, QRect, QSize

from iPhoto.gui.ui.window_manager import FramelessWindowManager


class _FakeScreen:
    def __init__(self, rect: QRect, dpr: float) -> None:
        self._rect = rect
        self._dpr = dpr

    def availableGeometry(self) -> QRect:  # noqa: N802 - Qt naming
        return QRect(self._rect)

    def devicePixelRatio(self) -> float:  # noqa: N802 - Qt naming
        return self._dpr


def test_clamp_size_to_available_limits_by_screen() -> None:
    """Window size should never exceed the target screen's available area."""

    current = QSize(6000, 4000)
    clamped = FramelessWindowManager._clamp_size_to_available(current, 1920, 1080)

    assert clamped.width() == 1880
    assert clamped.height() == 1040


def test_apply_screen_change_fix_rescales_and_repositions() -> None:
    """Moving to a denser screen should shrink and pull the window back on-screen."""

    manager = FramelessWindowManager.__new__(FramelessWindowManager)
    manager._geometry_fix_in_progress = False
    manager._last_screen_dpr = 1.0

    frame_rect = QRect(5000, 5000, 2200, 1600)

    window = MagicMock()
    window.size.return_value = QSize(2200, 1600)
    window.frameGeometry.return_value = frame_rect
    window.isFullScreen.return_value = False
    window.isMaximized.return_value = False
    manager._window = window

    snap_helper = MagicMock()
    snap_helper.is_snapped.return_value = False
    manager._snap_helper = snap_helper

    target_screen = _FakeScreen(QRect(0, 0, 2560, 1440), dpr=2.0)
    manager._apply_screen_change_fix(1.0, target_screen)

    window.resize.assert_called_once_with(QSize(1100, 800))

    # ``QWidget.move`` is overloaded in Qt and may be invoked either as
    # ``move(QPoint)`` or ``move(x, y)`` depending on binding/runtime details.
    # Accept both call signatures to keep the regression test platform-stable.
    window.move.assert_called_once()
    args, _ = window.move.call_args
    assert args in ((QPoint(20, 20),), (20, 20))
    assert manager._last_screen_dpr == 2.0


# ---------------------------------------------------------------------------
# _try_start_system_move – Wayland drag delegation
# ---------------------------------------------------------------------------

def _make_manager_with_window(window_handle=None) -> FramelessWindowManager:
    """Create a bare FramelessWindowManager instance suitable for unit tests."""
    manager = FramelessWindowManager.__new__(FramelessWindowManager)
    window = MagicMock()
    window.windowHandle.return_value = window_handle
    manager._window = window
    return manager


def test_try_start_system_move_on_wayland_calls_start_system_move() -> None:
    """startSystemMove() is called when the platform is 'wayland'."""
    handle = MagicMock()
    handle.startSystemMove = MagicMock()
    manager = _make_manager_with_window(handle)

    app = MagicMock()
    app.platformName.return_value = "wayland"

    with patch("iPhoto.gui.ui.window_manager.QApplication") as mock_app_cls:
        mock_app_cls.instance.return_value = app
        result = manager._try_start_system_move()

    assert result is True
    handle.startSystemMove.assert_called_once()


def test_try_start_system_move_on_xcb_returns_false() -> None:
    """startSystemMove() is NOT called on X11 (xcb platform)."""
    handle = MagicMock()
    manager = _make_manager_with_window(handle)

    app = MagicMock()
    app.platformName.return_value = "xcb"

    with patch("iPhoto.gui.ui.window_manager.QApplication") as mock_app_cls:
        mock_app_cls.instance.return_value = app
        result = manager._try_start_system_move()

    assert result is False
    handle.startSystemMove.assert_not_called()


def test_try_start_system_move_on_windows_returns_false() -> None:
    """startSystemMove() is NOT called on Windows ('windows' platform)."""
    handle = MagicMock()
    manager = _make_manager_with_window(handle)

    app = MagicMock()
    app.platformName.return_value = "windows"

    with patch("iPhoto.gui.ui.window_manager.QApplication") as mock_app_cls:
        mock_app_cls.instance.return_value = app
        result = manager._try_start_system_move()

    assert result is False
    handle.startSystemMove.assert_not_called()


def test_try_start_system_move_on_macos_returns_false() -> None:
    """startSystemMove() is NOT called on macOS ('cocoa' platform)."""
    handle = MagicMock()
    manager = _make_manager_with_window(handle)

    app = MagicMock()
    app.platformName.return_value = "cocoa"

    with patch("iPhoto.gui.ui.window_manager.QApplication") as mock_app_cls:
        mock_app_cls.instance.return_value = app
        result = manager._try_start_system_move()

    assert result is False
    handle.startSystemMove.assert_not_called()


def test_try_start_system_move_no_window_handle_returns_false() -> None:
    """Returns False gracefully when the window has no QWindow handle yet."""
    manager = _make_manager_with_window(window_handle=None)

    app = MagicMock()
    app.platformName.return_value = "wayland"

    with patch("iPhoto.gui.ui.window_manager.QApplication") as mock_app_cls:
        mock_app_cls.instance.return_value = app
        result = manager._try_start_system_move()

    assert result is False


def test_try_start_system_move_no_app_returns_false() -> None:
    """Returns False gracefully when QApplication has no instance."""
    manager = _make_manager_with_window()

    with patch("iPhoto.gui.ui.window_manager.QApplication") as mock_app_cls:
        mock_app_cls.instance.return_value = None
        result = manager._try_start_system_move()

    assert result is False
