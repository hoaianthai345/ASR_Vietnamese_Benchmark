from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def _lcs_pairs(a: Sequence[str], b: Sequence[str]) -> List[Tuple[int, int]]:
    # Returns one LCS alignment as index pairs (i, j) in increasing order.
    n = len(a)
    m = len(b)
    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ai = a[i]
        row = dp[i]
        row_next = dp[i + 1]
        for j in range(m - 1, -1, -1):
            if ai == b[j]:
                row[j] = 1 + row_next[j + 1]
            else:
                row[j] = row_next[j] if row_next[j] >= row[j + 1] else row[j + 1]

    pairs: List[Tuple[int, int]] = []
    i = 0
    j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j))
            i += 1
            j += 1
            continue
        if dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def lcs_merge_tokens(
    prev: Sequence[str],
    cur: Sequence[str],
    *,
    tail_window_tokens: int = 50,
) -> List[str]:
    """
    Merge two token sequences using an LCS-based overlap join.

    Rationale: in streaming, the current chunk often repeats / revises the tail of the
    previous hypothesis. We anchor the merge inside the tail of `prev` and then let
    `cur` override from the earliest aligned point.
    """
    if not prev:
        return list(cur)
    if not cur:
        return list(prev)

    pairs = _lcs_pairs(prev, cur)
    if not pairs:
        return list(prev) + list(cur)

    tail_start = max(0, len(prev) - max(1, tail_window_tokens))
    tail_pairs = [(i, j) for (i, j) in pairs if i >= tail_start]
    if tail_pairs:
        # Merge as early as possible in the current chunk while anchoring in the prev tail.
        i_anchor, j_anchor = min(tail_pairs, key=lambda ij: ij[1])
        return list(prev[:i_anchor]) + list(cur[j_anchor:])

    # Fallback: keep stable prefix of prev and append cur after the last aligned token.
    i_last, j_last = pairs[-1]
    return list(prev[: i_last + 1]) + list(cur[j_last + 1 :])


def _edit_distance_words(a: Sequence[str], b: Sequence[str]) -> int:
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


@dataclass(frozen=True)
class StabilityMetrics:
    # Sum of edit distances between successive partials, normalized by final length.
    edit_overhead: float
    # Fraction of steps with any change at all.
    change_rate: float
    # Average word-level edit distance between successive partials.
    avg_step_edits: float


def compute_stability(partials: Iterable[str]) -> StabilityMetrics:
    # partials: incremental hypotheses (strings) after each chunk merge.
    toks: List[List[str]] = [p.split() for p in partials]
    if not toks:
        return StabilityMetrics(edit_overhead=0.0, change_rate=0.0, avg_step_edits=0.0)

    total_edits = 0
    changed = 0
    steps = 0
    for prev, cur in zip(toks, toks[1:]):
        d = _edit_distance_words(prev, cur)
        total_edits += d
        changed += 1 if d > 0 else 0
        steps += 1

    final_len = max(1, len(toks[-1]))
    avg_step_edits = float(total_edits) / float(max(1, steps))
    edit_overhead = float(total_edits) / float(final_len)
    change_rate = float(changed) / float(max(1, steps))
    return StabilityMetrics(
        edit_overhead=edit_overhead,
        change_rate=change_rate,
        avg_step_edits=avg_step_edits,
    )

