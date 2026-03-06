"""Tests for Linux-specific bug fixes in gallery view, back button, and cluster navigation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iPhoto.gui.coordinators.navigation_coordinator import NavigationCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coordinator(
    *,
    current_album_root: Path | None = None,
    gallery_active: bool = True,
    library_root: Path | None = None,
) -> NavigationCoordinator:
    """Build a NavigationCoordinator with lightweight mocks."""

    sidebar = MagicMock()
    router = MagicMock()
    router.is_gallery_view_active.return_value = gallery_active

    facade = MagicMock()
    if current_album_root is not None:
        facade.current_album.root.resolve.return_value = current_album_root.resolve()
    else:
        facade.current_album = None

    context = MagicMock()
    if library_root is not None:
        context.library.root.return_value = library_root
    else:
        context.library.root.return_value = None

    album_service = MagicMock()
    asset_vm = MagicMock()
    event_bus = MagicMock()

    coord = NavigationCoordinator(
        sidebar=sidebar,
        router=router,
        album_service=album_service,
        asset_vm=asset_vm,
        event_bus=event_bus,
        context=context,
        facade=facade,
    )
    return coord


# ---------------------------------------------------------------------------
# Bug 3: Cluster gallery back button should disappear on view switch
# ---------------------------------------------------------------------------

class TestClusterGalleryModeCleanup:
    """Verify _exit_cluster_gallery_mode is called by all navigation methods."""

    def test_open_album_exits_cluster_mode(self, tmp_path: Path) -> None:
        album = tmp_path / "Paris"
        album.mkdir()
        coord = _make_coordinator(library_root=tmp_path)
        coord._in_cluster_gallery = True

        gallery_page = MagicMock()
        coord._router.gallery_page.return_value = gallery_page
        coord._facade.open_album.return_value = MagicMock(root=album)

        coord.open_album(album)

        assert coord._in_cluster_gallery is False
        gallery_page.set_cluster_gallery_mode.assert_called_with(False)

    def test_open_all_photos_exits_cluster_mode(self, tmp_path: Path) -> None:
        coord = _make_coordinator(library_root=tmp_path)
        coord._in_cluster_gallery = True

        gallery_page = MagicMock()
        coord._router.gallery_page.return_value = gallery_page

        coord.open_all_photos()

        assert coord._in_cluster_gallery is False
        gallery_page.set_cluster_gallery_mode.assert_called_with(False)

    def test_open_recently_deleted_exits_cluster_mode(self, tmp_path: Path) -> None:
        coord = _make_coordinator(library_root=tmp_path)
        coord._in_cluster_gallery = True

        gallery_page = MagicMock()
        coord._router.gallery_page.return_value = gallery_page
        coord._context.library.ensure_deleted_directory.return_value = tmp_path / ".deleted"
        coord._facade.open_album.return_value = None

        coord.open_recently_deleted()

        assert coord._in_cluster_gallery is False
        gallery_page.set_cluster_gallery_mode.assert_called_with(False)

    def test_filtered_collection_exits_cluster_mode(self, tmp_path: Path) -> None:
        coord = _make_coordinator(library_root=tmp_path)
        coord._in_cluster_gallery = True

        gallery_page = MagicMock()
        coord._router.gallery_page.return_value = gallery_page

        coord._open_filtered_collection("Favorites", is_favorite=True)

        assert coord._in_cluster_gallery is False
        gallery_page.set_cluster_gallery_mode.assert_called_with(False)

    def test_open_location_view_exits_cluster_mode(self, tmp_path: Path) -> None:
        coord = _make_coordinator(library_root=tmp_path)
        coord._in_cluster_gallery = True

        gallery_page = MagicMock()
        coord._router.gallery_page.return_value = gallery_page
        coord._context.library.get_geotagged_assets.return_value = []

        coord.open_location_view()

        assert coord._in_cluster_gallery is False
        gallery_page.set_cluster_gallery_mode.assert_called_with(False)

    def test_albums_dashboard_exits_cluster_mode(self) -> None:
        coord = _make_coordinator()
        coord._in_cluster_gallery = True

        gallery_page = MagicMock()
        coord._router.gallery_page.return_value = gallery_page

        coord._handle_static_node("Albums")

        assert coord._in_cluster_gallery is False
        gallery_page.set_cluster_gallery_mode.assert_called_with(False)

    def test_exit_cluster_gallery_noop_when_not_in_cluster(self) -> None:
        coord = _make_coordinator()
        coord._in_cluster_gallery = False

        coord._exit_cluster_gallery_mode()

        assert coord._in_cluster_gallery is False
        coord._router.gallery_page.assert_not_called()

    def test_return_to_map_uses_exit_helper(self, tmp_path: Path) -> None:
        coord = _make_coordinator(library_root=tmp_path)
        coord._in_cluster_gallery = True

        gallery_page = MagicMock()
        coord._router.gallery_page.return_value = gallery_page

        coord.return_to_map_from_cluster_gallery()

        assert coord._in_cluster_gallery is False
        gallery_page.set_cluster_gallery_mode.assert_called_with(False)
        coord._router.show_map.assert_called_once()
