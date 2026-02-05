from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass(frozen=True)
class ChunkConfig:
    sample_rate: int
    chunk_ms: int
    overlap_ms: int
    lookahead_ms: int


def iter_chunks(wav: np.ndarray, cfg: ChunkConfig) -> Iterator[np.ndarray]:
    if wav.ndim != 1:
        raise ValueError("wav must be mono 1D")
    sr = cfg.sample_rate
    chunk = int(sr * cfg.chunk_ms / 1000)
    overlap = int(sr * cfg.overlap_ms / 1000)
    lookahead = int(sr * cfg.lookahead_ms / 1000)

    step = max(1, chunk - overlap)
    n = wav.shape[0]

    start = 0
    while start < n:
        end = min(n, start + chunk + lookahead)
        yield wav[start:end]
        start += step
