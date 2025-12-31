from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DiarizationSegment:
    start: str
    end: str
    speaker: str
    transcript: str | None


class DiarizerBackend(Protocol):
    name: str

    def process_directory(self, norm_dir, out_dir) -> None:
        ...

