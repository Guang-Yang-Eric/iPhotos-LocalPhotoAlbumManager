"""Regression tests for immersive fullscreen edge-to-edge layout handling."""

from __future__ import annotations

from types import SimpleNamespace

from iPhoto.gui.ui.controllers.edit_fullscreen_manager import EditFullscreenManager
from iPhoto.gui.ui.window_manager import FramelessWindowManager


class _FakeMargins:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def left(self) -> int:
        return self._left

    def top(self) -> int:
        return self._top

    def right(self) -> int:
        return self._right

    def bottom(self) -> int:
        return self._bottom


class _FakeLayout:
    def __init__(self, margins: tuple[int, int, int, int]) -> None:
        self.current = margins

    def contentsMargins(self) -> _FakeMargins:  # noqa: N802 - Qt naming
        left, top, right, bottom = self.current
        return _FakeMargins(left, top, right, bottom)

    def setContentsMargins(self, left: int, top: int, right: int, bottom: int) -> None:  # noqa: N802 - Qt naming
        self.current = (left, top, right, bottom)


class _FakeSplitter:
    def __init__(self, handle_width: int) -> None:
        self._handle_width = handle_width

    def handleWidth(self) -> int:  # noqa: N802 - Qt naming
        return self._handle_width

    def setHandleWidth(self, width: int) -> None:  # noqa: N802 - Qt naming
        self._handle_width = width


def test_window_manager_content_layout_restores_after_fullscreen() -> None:
    manager = FramelessWindowManager.__new__(FramelessWindowManager)
    manager._ui = SimpleNamespace(
        right_panel_layout=_FakeLayout((8, 8, 8, 8)),
        splitter=_FakeSplitter(4),
    )
    manager._right_panel_margins_before = None
    manager._splitter_handle_width_before = None

    manager._apply_edge_to_edge_content_layout()
    assert manager._ui.right_panel_layout.current == (0, 0, 0, 0)
    assert manager._ui.splitter.handleWidth() == 0

    manager._restore_content_layout()
    assert manager._ui.right_panel_layout.current == (8, 8, 8, 8)
    assert manager._ui.splitter.handleWidth() == 4


def test_edit_fullscreen_content_layout_restores_after_exit() -> None:
    manager = EditFullscreenManager.__new__(EditFullscreenManager)
    manager._ui = SimpleNamespace(
        right_panel_layout=_FakeLayout((8, 8, 8, 8)),
        splitter=_FakeSplitter(5),
    )
    manager._fullscreen_right_panel_margins = None
    manager._fullscreen_splitter_handle_width = None

    manager._apply_edge_to_edge_content_layout()
    assert manager._ui.right_panel_layout.current == (0, 0, 0, 0)
    assert manager._ui.splitter.handleWidth() == 0

    manager._restore_content_layout()
    assert manager._ui.right_panel_layout.current == (8, 8, 8, 8)
    assert manager._ui.splitter.handleWidth() == 5
