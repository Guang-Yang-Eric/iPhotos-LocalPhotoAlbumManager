"""Tests for fullscreen layout adjustments in FramelessWindowManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from iPhoto.gui.ui.window_manager import FramelessWindowManager


@pytest.fixture(autouse=True)
def _ensure_qapp():
    """Make sure a QApplication instance exists for widget creation."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield


def _make_stub_ui():
    """Build a minimal Ui_MainWindow-like stub with enough attributes."""
    ui = MagicMock()

    # --- right_panel with a real QWidget and layout ---
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(8, 8, 8, 8)
    ui.right_panel = right_panel

    # --- window_shell ---
    window_shell = QWidget()
    window_shell.setStyleSheet("")
    ui.window_shell = window_shell

    # --- player_container / player_stack ---
    player_container = QWidget()
    player_container.setStyleSheet("")
    ui.player_container = player_container

    player_stack = QWidget()
    player_stack.setStyleSheet("")
    ui.player_stack = player_stack

    # --- splitter ---
    splitter = MagicMock()
    splitter.sizes.return_value = [200, 800]
    splitter.signalsBlocked.return_value = False
    ui.splitter = splitter

    # --- simple stubs for widgets that need to be visible/hidden ---
    ui.window_chrome = QWidget()
    ui.menu_bar_container = QWidget()
    ui.menu_bar = MagicMock()
    ui.status_bar = QWidget()
    ui.sidebar = QWidget()
    ui.detail_chrome_container = QWidget()
    ui.filmstrip_view = QWidget()
    ui.title_bar = QWidget()
    ui.window_title_label = MagicMock()
    ui.minimize_button = MagicMock()
    ui.close_button = MagicMock()
    ui.fullscreen_button = MagicMock()

    # --- image_viewer / video_area ---
    ui.image_viewer = MagicMock()
    ui.video_area = MagicMock()
    ui.video_area.controls_enabled.return_value = False

    # --- view_stack ---
    ui.view_stack = MagicMock()
    detail_page = QWidget()
    ui.detail_page = detail_page
    ui.view_stack.currentWidget.return_value = detail_page

    # --- badge host ---
    ui.badge_host = QWidget()
    ui.live_badge = MagicMock()

    # --- toggle_filmstrip_action ---
    ui.toggle_filmstrip_action = MagicMock()

    # --- resize bits ---
    ui.resize_indicator = None
    ui.size_grip = None

    return ui


def _make_manager():
    """Create a FramelessWindowManager with a mock controller (no edit controller)."""
    window = QMainWindow()
    ui = _make_stub_ui()
    window.setCentralWidget(ui.window_shell)

    manager = FramelessWindowManager(window, ui)
    controller = MagicMock()
    controller.suspend_playback_for_transition.return_value = False
    controller.prepare_fullscreen_asset.return_value = True
    controller.is_edit_view_active.return_value = False
    # Ensure _edit_controller() returns None so exit_fullscreen runs
    # through the main path rather than short-circuiting into edit mode.
    controller.edit_controller = None
    manager.set_controller(controller)
    return manager, ui, window


def test_enter_fullscreen_zeroes_right_panel_margins():
    """Right panel layout margins should be (0, 0, 0, 0) after entering fullscreen."""
    manager, ui, _window = _make_manager()

    layout = ui.right_panel.layout()
    m = layout.contentsMargins()
    assert (m.left(), m.top(), m.right(), m.bottom()) == (8, 8, 8, 8)

    manager.enter_fullscreen()

    m = layout.contentsMargins()
    assert (m.left(), m.top(), m.right(), m.bottom()) == (0, 0, 0, 0)
    manager.cleanup()


def test_exit_fullscreen_restores_right_panel_margins():
    """Right panel margins should be restored to (8, 8, 8, 8) after exiting fullscreen."""
    manager, ui, _window = _make_manager()

    manager.enter_fullscreen()
    layout = ui.right_panel.layout()
    m = layout.contentsMargins()
    assert (m.left(), m.top(), m.right(), m.bottom()) == (0, 0, 0, 0)

    manager.exit_fullscreen()
    m = layout.contentsMargins()
    assert (m.left(), m.top(), m.right(), m.bottom()) == (8, 8, 8, 8)
    manager.cleanup()


def test_immersive_backdrop_includes_right_panel():
    """Right panel background should be black during fullscreen and restored on exit."""
    manager, ui, _window = _make_manager()
    original_style = ui.right_panel.styleSheet()

    manager.enter_fullscreen()
    assert "background-color: #000000" in ui.right_panel.styleSheet()

    manager.exit_fullscreen()
    assert ui.right_panel.styleSheet() == original_style
    manager.cleanup()
