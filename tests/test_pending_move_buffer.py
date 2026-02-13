import importlib.util
from pathlib import Path
import sys

from iPhoto.application.dtos import AssetDTO


def _load_pending_move_buffer():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src/iPhoto/gui/viewmodels/pending_move_buffer.py"
    spec = importlib.util.spec_from_file_location("test_pending_move_buffer_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_pending_move_buffer_module"] = module
    spec.loader.exec_module(module)
    return module


PendingMove = _load_pending_move_buffer().PendingMove


def test_pending_move_tracks_destination_state():
    dto = AssetDTO(
        id="id1",
        abs_path=Path("/library/src.jpg"),
        rel_path=Path("Album/src.jpg"),
        media_type="image",
        created_at=None,
        width=1,
        height=1,
        duration=0.0,
        size_bytes=1,
        metadata={},
        is_favorite=False,
    )
    pending = PendingMove(
        dto=dto,
        source_abs=Path("/library/src.jpg"),
        destination_root=Path("/library"),
        destination_album_path="Album2",
        destination_abs=Path("/library/Album2/src.jpg"),
        destination_rel=Path("Album2/src.jpg"),
        is_delete=False,
    )

    assert pending.destination_album_path == "Album2"
    assert pending.destination_rel.as_posix() == "Album2/src.jpg"
