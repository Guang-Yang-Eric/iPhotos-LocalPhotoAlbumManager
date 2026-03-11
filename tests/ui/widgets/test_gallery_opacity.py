"""Tests that gallery widgets block the WA_TranslucentBackground cascade.

The main window uses ``WA_TranslucentBackground`` for frameless rounded-corner
chrome.  When QRhiWidget instances exist elsewhere in the widget tree (e.g. on
the detail page), Qt activates a GL-based texture compositor for the entire
window.  Unless gallery widgets explicitly block the cascade, the compositor
treats their backing stores as ARGB textures whose async upload can race with
scroll repaints, producing visible tearing on Linux.
"""

import pytest

pytest.importorskip("PySide6", reason="PySide6 is required for GUI tests", exc_type=ImportError)
pytest.importorskip("PySide6.QtWidgets", reason="Qt widgets not available", exc_type=ImportError)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from iPhoto.gui.ui.widgets.gallery_grid_view import GalleryGridView
from iPhoto.gui.ui.widgets.gallery_page import GalleryPageWidget


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_gallery_page_blocks_translucent_cascade(qapp: QApplication) -> None:
    """GalleryPageWidget must block WA_TranslucentBackground from the window."""
    page = GalleryPageWidget()
    assert not page.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert page.autoFillBackground()


def test_gallery_grid_view_blocks_translucent_cascade(qapp: QApplication) -> None:
    """GalleryGridView and its viewport must block WA_TranslucentBackground."""
    view = GalleryGridView()
    assert not view.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    vp = view.viewport()
    assert not vp.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert vp.autoFillBackground()
