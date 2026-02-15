"""Tests for P2 use cases: ManageTrash, AggregateGeoData, WatchFilesystem, ExportAssets, ApplyEdit."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from iPhoto.application.use_cases.manage_trash import (
    ManageTrashUseCase,
    ManageTrashRequest,
)
from iPhoto.application.use_cases.aggregate_geo_data import (
    AggregateGeoDataUseCase,
    AggregateGeoDataRequest,
)
from iPhoto.application.use_cases.watch_filesystem import (
    WatchFilesystemUseCase,
    WatchFilesystemRequest,
)
from iPhoto.application.use_cases.export_assets import (
    ExportAssetsUseCase,
    ExportAssetsRequest,
)
from iPhoto.application.use_cases.apply_edit import (
    ApplyEditUseCase,
    ApplyEditRequest,
)


# ===========================================================================
# ManageTrashUseCase
# ===========================================================================

class TestManageTrashUseCase:
    def test_no_trash_manager_returns_error(self):
        uc = ManageTrashUseCase(trash_manager=None)
        response = uc.execute(ManageTrashRequest())
        assert response.success is False
        assert "not configured" in response.error

    def test_cleanup_action(self):
        manager = MagicMock()
        manager.cleanup_deleted_index.return_value = 5

        uc = ManageTrashUseCase(trash_manager=manager)
        response = uc.execute(ManageTrashRequest(action="cleanup"))

        assert response.success is True
        assert response.cleaned_count == 5
        manager.cleanup_deleted_index.assert_called_once()

    def test_ensure_action(self):
        manager = MagicMock()
        manager.ensure_deleted_directory.return_value = Path("/lib/.deleted")

        uc = ManageTrashUseCase(trash_manager=manager)
        response = uc.execute(ManageTrashRequest(action="ensure"))

        assert response.success is True
        assert response.trash_path == "/lib/.deleted"
        manager.ensure_deleted_directory.assert_called_once()

    def test_get_path_action(self):
        manager = MagicMock()
        manager.deleted_directory.return_value = Path("/lib/.deleted")

        uc = ManageTrashUseCase(trash_manager=manager)
        response = uc.execute(ManageTrashRequest(action="get_path"))

        assert response.success is True
        assert response.trash_path == "/lib/.deleted"
        manager.deleted_directory.assert_called_once()

    def test_unknown_action_returns_error(self):
        manager = MagicMock()

        uc = ManageTrashUseCase(trash_manager=manager)
        response = uc.execute(ManageTrashRequest(action="nope"))

        assert response.success is False
        assert "Unknown action" in response.error


# ===========================================================================
# AggregateGeoDataUseCase
# ===========================================================================

class TestAggregateGeoDataUseCase:
    def test_no_geo_aggregator_returns_error(self):
        uc = AggregateGeoDataUseCase(geo_aggregator=None)
        response = uc.execute(AggregateGeoDataRequest())
        assert response.success is False
        assert "not configured" in response.error

    def test_successful_aggregation(self):
        aggregator = MagicMock()
        aggregator.get_geotagged_assets.return_value = [
            {"id": "a1", "lat": 35.0, "lon": 139.0},
            {"id": "a2", "lat": 40.0, "lon": -74.0},
        ]

        uc = AggregateGeoDataUseCase(geo_aggregator=aggregator)
        response = uc.execute(AggregateGeoDataRequest())

        assert response.success is True
        assert response.total_count == 2
        assert len(response.assets) == 2
        aggregator.get_geotagged_assets.assert_called_once()


# ===========================================================================
# WatchFilesystemUseCase
# ===========================================================================

class TestWatchFilesystemUseCase:
    def test_no_watcher_returns_error(self):
        uc = WatchFilesystemUseCase(watcher=None)
        response = uc.execute(WatchFilesystemRequest())
        assert response.success is False
        assert "not configured" in response.error

    def test_pause_action(self):
        watcher = MagicMock()

        uc = WatchFilesystemUseCase(watcher=watcher)
        response = uc.execute(WatchFilesystemRequest(action="pause"))

        assert response.success is True
        assert response.is_paused is True
        watcher.pause_watcher.assert_called_once()

    def test_resume_action(self):
        watcher = MagicMock()

        uc = WatchFilesystemUseCase(watcher=watcher)
        response = uc.execute(WatchFilesystemRequest(action="resume"))

        assert response.success is True
        assert response.is_paused is False
        watcher.resume_watcher.assert_called_once()


# ===========================================================================
# ExportAssetsUseCase
# ===========================================================================

class TestExportAssetsUseCase:
    def test_no_source_paths_returns_zero(self):
        export_fn = MagicMock()

        uc = ExportAssetsUseCase(export_fn=export_fn)
        response = uc.execute(ExportAssetsRequest(source_paths=[]))

        assert response.success is True
        assert response.exported_count == 0
        export_fn.assert_not_called()

    def test_successful_export(self):
        export_fn = MagicMock(return_value=True)

        uc = ExportAssetsUseCase(export_fn=export_fn)
        response = uc.execute(ExportAssetsRequest(
            source_paths=[Path("/photos/a.jpg"), Path("/photos/b.jpg")],
            export_root=Path("/out"),
            library_root=Path("/lib"),
        ))

        assert response.success is True
        assert response.exported_count == 2
        assert response.failed_count == 0
        assert export_fn.call_count == 2

    def test_mixed_success_and_failure(self):
        export_fn = MagicMock(side_effect=[True, False, True])

        uc = ExportAssetsUseCase(export_fn=export_fn)
        response = uc.execute(ExportAssetsRequest(
            source_paths=[Path("/a.jpg"), Path("/b.jpg"), Path("/c.jpg")],
            export_root=Path("/out"),
            library_root=Path("/lib"),
        ))

        assert response.success is True
        assert response.exported_count == 2
        assert response.failed_count == 1
        assert "/b.jpg" in response.failed_paths


# ===========================================================================
# ApplyEditUseCase
# ===========================================================================

class TestApplyEditUseCase:
    def test_render_returns_none_gives_error(self):
        render_fn = MagicMock(return_value=None)

        uc = ApplyEditUseCase(render_fn=render_fn)
        response = uc.execute(ApplyEditRequest(asset_path=Path("/photo.jpg")))

        assert response.success is False
        assert "no result" in response.error.lower()

    def test_successful_render(self, tmp_path):
        rendered = MagicMock()
        render_fn = MagicMock(return_value=rendered)
        output = tmp_path / "out.jpg"

        uc = ApplyEditUseCase(render_fn=render_fn)
        response = uc.execute(ApplyEditRequest(
            asset_path=Path("/photo.jpg"),
            output_path=output,
        ))

        assert response.success is True
        assert response.output_path == str(output)
        render_fn.assert_called_once_with(Path("/photo.jpg"))
        rendered.save.assert_called_once_with(str(output), "JPG", 100)
