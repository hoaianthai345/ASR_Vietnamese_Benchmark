from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol
import numpy as np


@dataclass(frozen=True)
class DecodeResult:
    text: str
    rtf: float  # real-time factor approximation (compute_time / audio_time)


class ASRModel(Protocol):
    def transcribe_batch(self, wavs: List[np.ndarray], sample_rate: int) -> List[DecodeResult]:
        ...
