from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ASRSegmentTimes:
    start: float
    end: float


@dataclass(frozen=True)
class ASRResult:
    text: str
    times: ASRSegmentTimes | None = None
    raw: Any | None = None


class ASRBackend(Protocol):
    name: str

    def transcribe_batch(self, paths: list[str]) -> list[ASRResult]:
        ...

