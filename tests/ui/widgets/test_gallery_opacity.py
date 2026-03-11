"""Tests that gallery widgets block the WA_TranslucentBackground cascade.

The main window uses ``WA_TranslucentBackground`` for frameless rounded-corner
chrome.  When QRhiWidget instances exist elsewhere in the widget tree (e.g. on
the detail page), Qt activates a GL-based texture compositor for the entire
window.  Unless gallery widgets explicitly block the cascade, the compositor
treats their backing stores as ARGB textures whose async upload can race with
scroll repaints, producing visible tearing on Linux.
"""

import sys

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


def test_gallery_grid_view_no_double_painter(qapp: QApplication) -> None:
    """GalleryGridView must NOT override paintEvent.

    A second QPainter on the viewport after the base class has flushed its
    painter corrupts the GL backing texture when the texture-based compositor
    is active (triggered by QRhiWidget on the detail page).  Verify that
    ``paintEvent`` is NOT overridden in GalleryGridView.
    """
    # If GalleryGridView defines its own paintEvent, it would shadow the base
    # class method. Verify it does not.
    assert "paintEvent" not in GalleryGridView.__dict__


@pytest.mark.skipif(sys.platform != "linux", reason="GL viewport is Linux-only")
def test_gallery_grid_view_uses_gl_viewport_on_linux(qapp: QApplication) -> None:
    """On Linux the viewport must be a QOpenGLWidget for FBO double-buffering.

    The main window uses ``WA_TranslucentBackground`` which forces an ARGB
    backing store.  A standard QWidget viewport shares this backing store and
    the compositor can read it mid-paint, causing garbled pixel blocks.
    ``QOpenGLWidget`` renders into its own FBO so only completed frames are
    submitted to the compositor.
    """
    try:
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
    except ImportError:
        pytest.skip("QOpenGLWidget not available")

    view = GalleryGridView()
    assert isinstance(view.viewport(), QOpenGLWidget)


def test_gallery_grid_view_schedules_viewport_repaint_on_data_change(qapp: QApplication) -> None:
    """setModel must wire dataChanged → _schedule_viewport_repaint on Linux."""
    from unittest.mock import patch

    from PySide6.QtGui import QStandardItemModel, QStandardItem

    view = GalleryGridView()
    model = QStandardItemModel()
    for i in range(5):
        model.appendRow(QStandardItem(f"item-{i}"))

    with patch(
        "iPhoto.gui.ui.widgets.gallery_grid_view._IS_LINUX", True
    ):
        view.setModel(model)

    # Verify the connection fires _schedule_viewport_repaint.
    with patch.object(view, "_schedule_viewport_repaint") as mock_repaint:
        idx = model.index(0, 0)
        model.setData(idx, "changed")
        mock_repaint.assert_called()


def test_schedule_viewport_repaint_posts_deferred_update(qapp: QApplication) -> None:
    """_schedule_viewport_repaint must post a deferred viewport().update()."""
    from unittest.mock import patch

    view = GalleryGridView()
    vp = view.viewport()

    with patch("iPhoto.gui.ui.widgets.gallery_grid_view.QTimer") as mock_timer:
        view._schedule_viewport_repaint()
        mock_timer.singleShot.assert_called_once_with(0, vp.update)
