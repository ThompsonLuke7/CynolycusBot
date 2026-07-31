"""Immutable, streaming registration of source artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


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


__all__ = ["SourceArtifact", "register_artifact"]
