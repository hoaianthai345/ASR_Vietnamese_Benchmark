from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
from jiwer import compute_measures

from vnasrbench.utils.textnorm import normalize_vi


@dataclass(frozen=True)
class WerResult:
    wer: float
    n_ref_words: int


def wer(refs: Sequence[str], hyps: Sequence[str]) -> WerResult:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have same length")

    n_ref = 0
    measures_total = {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
    for r, h in zip(refs, hyps):
        rn = normalize_vi(r)
        hn = normalize_vi(h)
        m = compute_measures(rn, hn)
        measures_total["substitutions"] += int(m["substitutions"])
        measures_total["deletions"] += int(m["deletions"])
        measures_total["insertions"] += int(m["insertions"])
        measures_total["hits"] += int(m["hits"])
        n_ref += len(rn.split()) if rn else 0

    denom = max(1, measures_total["hits"] + measures_total["substitutions"] + measures_total["deletions"])
    wer_value = (measures_total["substitutions"] + measures_total["deletions"] + measures_total["insertions"]) / denom
    return WerResult(wer=wer_value, n_ref_words=n_ref)


def paired_samples(refs: Sequence[str], hyps: Sequence[str]) -> List[Tuple[str, str]]:
    return list(zip(refs, hyps))
