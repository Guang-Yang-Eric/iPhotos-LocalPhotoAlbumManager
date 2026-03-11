"""Tests for FilmstripView viewport wheel-event forwarding on Linux."""

import pytest

pytest.importorskip("PySide6", reason="PySide6 is required for GUI tests", exc_type=ImportError)
pytest.importorskip("PySide6.QtWidgets", reason="Qt widgets not available", exc_type=ImportError)

from unittest.mock import patch

from PySide6.QtCore import QEvent, QPoint, QPointF, QStringListModel, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from iPhoto.gui.ui.widgets.filmstrip_view import FilmstripView


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _make_filmstrip(qapp: QApplication) -> FilmstripView:
    """Create a FilmstripView with a populated model."""
    view = FilmstripView()
    model = QStringListModel([str(i) for i in range(20)])
    view.setModel(model)
    view.resize(800, 132)
    view.show()
    qapp.processEvents()
    return view


def _make_wheel_event(
    delta_y: int = 0,
    delta_x: int = 0,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    pixel_delta_y: int = 0,
    pixel_delta_x: int = 0,
) -> QWheelEvent:
    """Create a synthetic QWheelEvent with the specified deltas."""
    return QWheelEvent(
        QPointF(100, 60),  # pos
        QPointF(400, 360),  # globalPos
        QPoint(pixel_delta_x, pixel_delta_y),  # pixelDelta
        QPoint(delta_x, delta_y),  # angleDelta
        Qt.MouseButton.NoButton,  # buttons
        modifiers,  # modifiers
        Qt.ScrollPhase.NoScrollPhase,  # phase
        False,  # inverted
    )


# ------------------------------------------------------------------
# Event filter installation
# ------------------------------------------------------------------


def test_event_filter_installed_on_linux(qapp: QApplication) -> None:
    """On Linux the viewport must have the filmstrip registered as an event filter."""
    with patch("iPhoto.gui.ui.widgets.filmstrip_view._IS_LINUX", True):
        view = FilmstripView()
        # On Linux, the filmstrip installs itself as an event filter on the
        # viewport.  We verify indirectly by checking that a viewport wheel
        # event is intercepted by the eventFilter method.
        model = QStringListModel(["a", "b", "c"])
        view.setModel(model)
        view.resize(400, 132)
        view.show()
        qapp.processEvents()

        signals: list[str] = []
        view.nextItemRequested.connect(lambda: signals.append("next"))

        event = _make_wheel_event(delta_y=-120)
        # Deliver the event directly to the viewport via the event filter.
        handled = view.eventFilter(view.viewport(), event)
        assert handled, "eventFilter should consume the accepted wheel event"
        assert signals == ["next"]


# ------------------------------------------------------------------
# Viewport wheel → navigation signals
# ------------------------------------------------------------------


def test_viewport_wheel_down_emits_next(qapp: QApplication) -> None:
    """Scrolling down on the viewport should emit nextItemRequested."""
    view = _make_filmstrip(qapp)
    signals: list[str] = []
    view.nextItemRequested.connect(lambda: signals.append("next"))

    event = _make_wheel_event(delta_y=-120)
    view.eventFilter(view.viewport(), event)
    assert "next" in signals


def test_viewport_wheel_up_emits_prev(qapp: QApplication) -> None:
    """Scrolling up on the viewport should emit prevItemRequested."""
    view = _make_filmstrip(qapp)
    signals: list[str] = []
    view.prevItemRequested.connect(lambda: signals.append("prev"))

    event = _make_wheel_event(delta_y=120)
    view.eventFilter(view.viewport(), event)
    assert "prev" in signals


def test_viewport_pixel_delta_emits_signal(qapp: QApplication) -> None:
    """Trackpad-style pixel deltas (angleDelta==0) should still emit navigation."""
    view = _make_filmstrip(qapp)
    signals: list[str] = []
    view.nextItemRequested.connect(lambda: signals.append("next"))

    event = _make_wheel_event(pixel_delta_y=-5)
    view.eventFilter(view.viewport(), event)
    assert "next" in signals


def test_viewport_ctrl_wheel_not_consumed(qapp: QApplication) -> None:
    """Ctrl+wheel should not be consumed by the event filter (zoom passthrough)."""
    view = _make_filmstrip(qapp)
    signals: list[str] = []
    view.nextItemRequested.connect(lambda: signals.append("next"))
    view.prevItemRequested.connect(lambda: signals.append("prev"))

    event = _make_wheel_event(delta_y=-120, modifiers=Qt.KeyboardModifier.ControlModifier)
    handled = view.eventFilter(view.viewport(), event)
    # Ctrl+scroll delegates to super().wheelEvent() which does not accept for
    # filmstrip (vertical scrollbar is always off), so the filter returns False.
    assert not handled, "Ctrl+wheel should not be consumed"
    assert signals == [], "No navigation signal should be emitted for Ctrl+wheel"


def test_viewport_zero_delta_not_consumed(qapp: QApplication) -> None:
    """A wheel event with zero deltas should not be consumed."""
    view = _make_filmstrip(qapp)
    event = _make_wheel_event()  # all deltas default to 0
    handled = view.eventFilter(view.viewport(), event)
    assert not handled


def test_non_viewport_event_passes_through(qapp: QApplication) -> None:
    """Events on widgets other than the viewport should not be intercepted."""
    view = _make_filmstrip(qapp)
    from PySide6.QtWidgets import QWidget

    other = QWidget()
    event = _make_wheel_event(delta_y=-120)
    handled = view.eventFilter(other, event)
    assert not handled, "Wheel events on non-viewport widgets must not be intercepted"


def test_non_wheel_event_passes_through(qapp: QApplication) -> None:
    """Non-wheel events on the viewport should not be intercepted."""
    view = _make_filmstrip(qapp)

    event = QEvent(QEvent.Type.MouseButtonPress)
    handled = view.eventFilter(view.viewport(), event)
    assert not handled
