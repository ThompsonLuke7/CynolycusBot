"""Immutable, streaming registration of source artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryFile
from collections.abc import Iterator
from typing import BinaryIO


_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class SourceArtifact:
    """The filesystem facts needed to register one immutable source."""

    path: Path
    uri: str
    sha256: str
    byte_size: int

    @property
    def source_path(self) -> Path:
        return self.path


def register_artifact(path: Path) -> SourceArtifact:
    """Hash ``path`` and count bytes without modifying or stat'ing it.

    Availability is a semantic property of an imported record.  This function
    deliberately uses neither filesystem modification time nor any other
    filesystem metadata as evidence.
    """

    source_path = Path(path)
    digest = sha256()
    byte_size = 0
    try:
        with source_path.open("rb") as source_file:
            while True:
                chunk = source_file.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                byte_size += len(chunk)
    except OSError as exc:
        raise FileNotFoundError(f"unable to read source artifact: {source_path}") from exc
    return SourceArtifact(
        path=source_path,
        uri=source_path.as_posix(),
        sha256=digest.hexdigest(),
        byte_size=byte_size,
    )


@contextmanager
def snapshot_artifact(path: Path) -> Iterator[tuple[SourceArtifact, BinaryIO]]:
    """Capture one immutable byte snapshot and expose it as a seekable stream.

    The importer parses this snapshot rather than reopening a mutable source
    path.  The caller still verifies the path hash after parsing so an append
    or replacement during the import is rejected instead of silently mixing
    generations.
    """

    source_path = Path(path)
    digest = sha256()
    byte_size = 0
    try:
        with source_path.open("rb") as source_file, TemporaryFile(mode="w+b") as snapshot:
            while True:
                chunk = source_file.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                byte_size += len(chunk)
                snapshot.write(chunk)
            artifact = SourceArtifact(
                path=source_path,
                uri=source_path.as_posix(),
                sha256=digest.hexdigest(),
                byte_size=byte_size,
            )
            snapshot.seek(0)
            yield artifact, snapshot
    except OSError as exc:
        raise FileNotFoundError(f"unable to read source artifact: {source_path}") from exc


__all__ = ["SourceArtifact", "register_artifact", "snapshot_artifact"]
