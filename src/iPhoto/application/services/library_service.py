import logging
from pathlib import Path
from typing import List, Optional

from iPhoto.domain.repositories import IAlbumRepository
from iPhoto.application.use_cases.create_album import CreateAlbumUseCase, CreateAlbumRequest
from iPhoto.application.use_cases.delete_album import DeleteAlbumUseCase, DeleteAlbumRequest
from iPhoto.application.services.parallel_scanner import ParallelScanner, ScanResult

class LibraryService:
    """Application Service for library-level operations."""
    
    def __init__(
        self,
        album_repo: IAlbumRepository,
        create_album_uc: CreateAlbumUseCase,
        delete_album_uc: DeleteAlbumUseCase,
        parallel_scanner: Optional[ParallelScanner] = None,
    ):
        self._album_repo = album_repo
        self._create_album_uc = create_album_uc
        self._delete_album_uc = delete_album_uc
        self._parallel_scanner = parallel_scanner
        self._logger = logging.getLogger(__name__)

    def create_album(self, path: Path, title: str = "") -> str:
        response = self._create_album_uc.execute(CreateAlbumRequest(path=path, title=title))
        if not response.success:
            raise ValueError(response.error)
        return response.album_id

    def delete_album(self, album_id: str) -> None:
        response = self._delete_album_uc.execute(DeleteAlbumRequest(album_id=album_id))
        if not response.success:
            raise ValueError(response.error)

    def scan_parallel(self, album_path: Path) -> Optional[ScanResult]:
        """Run a parallel scan of the given album path.

        Returns a :class:`ScanResult` when a :class:`ParallelScanner` has been
        configured, or ``None`` if no scanner is available.
        """
        if self._parallel_scanner is None:
            self._logger.warning("ParallelScanner not configured; skipping parallel scan")
            return None
        return self._parallel_scanner.scan(album_path)
