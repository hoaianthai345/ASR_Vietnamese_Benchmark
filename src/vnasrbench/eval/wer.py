from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

# jiwer API changed across versions; support both compute_measures (older) and process_words (newer).
try:  # pragma: no cover
    from jiwer import process_words as _process_words  # type: ignore

    def _measures(ref: str, hyp: str) -> dict:
        out = _process_words(ref, hyp)
        return {
            "substitutions": int(out.substitutions),
            "deletions": int(out.deletions),
            "insertions": int(out.insertions),
            "hits": int(out.hits),
        }

except Exception:  # pragma: no cover
    from jiwer import compute_measures as _compute_measures  # type: ignore

    def _measures(ref: str, hyp: str) -> dict:
        m = _compute_measures(ref, hyp)
        return {
            "substitutions": int(m["substitutions"]),
            "deletions": int(m["deletions"]),
            "insertions": int(m["insertions"]),
            "hits": int(m["hits"]),
        }

from vnasrbench.utils.textnorm import normalize_vi, normalize_cer_strict_vi


@dataclass(frozen=True)
class WerResult:
    wer: float
    n_ref_words: int


def wer(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    normalizer: Callable[[str], str] = normalize_vi,
) -> WerResult:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have same length")

    n_ref = 0
    measures_total = {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
    for r, h in zip(refs, hyps):
        rn = normalizer(r)
        hn = normalizer(h)
        m = _measures(rn, hn)
        measures_total["substitutions"] += int(m["substitutions"])
        measures_total["deletions"] += int(m["deletions"])
        measures_total["insertions"] += int(m["insertions"])
        measures_total["hits"] += int(m["hits"])
        n_ref += len(rn.split()) if rn else 0

    denom = max(1, measures_total["hits"] + measures_total["substitutions"] + measures_total["deletions"])
    wer_value = (measures_total["substitutions"] + measures_total["deletions"] + measures_total["insertions"]) / denom
    return WerResult(wer=wer_value, n_ref_words=n_ref)


@dataclass(frozen=True)
class CerResult:
    cer: float
    n_ref_chars: int


def _edit_distance_chars(a: Sequence[str], b: Sequence[str]) -> int:
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev_diag = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,  # deletion
                dp[j - 1] + 1,  # insertion
                prev_diag + cost,  # substitution
            )
            prev_diag = tmp
    return dp[m]


def cer(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    normalizer: Callable[[str], str] = normalize_cer_strict_vi,
) -> CerResult:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have same length")

    n_ref = 0
    total_edits = 0
    for r, h in zip(refs, hyps):
        rn = normalizer(r)
        hn = normalizer(h)
        total_edits += _edit_distance_chars(rn, hn)
        n_ref += len(rn)

    denom = max(1, n_ref)
    cer_value = float(total_edits) / float(denom)
    return CerResult(cer=cer_value, n_ref_chars=n_ref)


def paired_samples(refs: Sequence[str], hyps: Sequence[str]) -> List[Tuple[str, str]]:
    return list(zip(refs, hyps))
