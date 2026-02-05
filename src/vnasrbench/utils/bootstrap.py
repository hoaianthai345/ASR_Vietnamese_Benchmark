from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np

from vnasrbench.eval.wer import wer


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    low: float
    high: float


def bootstrap_wer_ci(
    pairs: Sequence[Tuple[str, str]],
    n_samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    n = len(pairs)
    if n == 0:
        raise ValueError("No samples to bootstrap")

    wers: List[float] = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        refs = [pairs[i][0] for i in idx]
        hyps = [pairs[i][1] for i in idx]
        wers.append(wer(refs, hyps).wer)

    arr = np.array(wers, dtype=np.float64)
    mean = float(arr.mean())
    low = float(np.quantile(arr, alpha / 2))
    high = float(np.quantile(arr, 1 - alpha / 2))
    return BootstrapCI(mean=mean, low=low, high=high)
