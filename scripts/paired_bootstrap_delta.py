from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from vnasrbench.eval.wer import wer


def _identity(x: str) -> str:
    return x


def _load_hypotheses(path: Path) -> Tuple[List[str], List[str], List[str]]:
    refs: List[str] = []
    off: List[str] = []
    stream: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # Prefer pre-normalized strict fields if present
            ref = row.get("ref_norm_strict", row.get("ref", ""))
            hyp_off = row.get("hyp_offline_norm_strict", row.get("hyp_offline", ""))
            hyp_stream = row.get("hyp_streaming_norm_strict", row.get("hyp_streaming", ""))
            if not ref or hyp_stream is None:
                continue
            refs.append(str(ref))
            off.append(str(hyp_off))
            stream.append(str(hyp_stream))
    return refs, off, stream


def _paired_bootstrap_delta(
    refs: List[str],
    off: List[str],
    stream: List[str],
    *,
    n_samples: int,
    seed: int,
    score_fn: Callable[[List[str], List[str]], float],
) -> Dict[str, float]:
    n = len(refs)
    if n == 0:
        raise ValueError("No samples available")
    if not (len(off) == n and len(stream) == n):
        raise ValueError("refs/off/stream lengths mismatch")

    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        r = [refs[j] for j in idx]
        o = [off[j] for j in idx]
        s = [stream[j] for j in idx]
        deltas[i] = score_fn(r, s) - score_fn(r, o)

    observed_delta = score_fn(refs, stream) - score_fn(refs, off)
    ci_low = float(np.quantile(deltas, 0.025))
    ci_high = float(np.quantile(deltas, 0.975))
    # One-sided p-value: probability that streaming is not worse than offline
    p_nonpositive = float(np.mean(deltas <= 0.0))

    return {
        "n_utts": float(n),
        "delta_wer_obs": float(observed_delta),
        "delta_wer_ci_low": ci_low,
        "delta_wer_ci_high": ci_high,
        "p_nonpositive": p_nonpositive,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_csv", default="results/delta_significance.csv")
    ap.add_argument("--filter", default="", help="Substring filter on run_id")
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows: List[Dict[str, float | str]] = []
    for hyp_path in sorted(runs_dir.glob("*/hypotheses.jsonl")):
        run_id = hyp_path.parent.name
        if args.filter and args.filter not in run_id:
            continue

        refs, off, stream = _load_hypotheses(hyp_path)
        # Need streaming hypotheses for paired test
        if len(stream) == 0:
            continue

        stats = _paired_bootstrap_delta(
            refs,
            off,
            stream,
            n_samples=args.n_samples,
            seed=args.seed,
            score_fn=lambda r, h: float(wer(r, h, normalizer=_identity).wer),
        )
        row: Dict[str, float | str] = {"run_id": run_id}
        row.update(stats)
        rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No eligible runs with hypotheses.jsonl + streaming hypotheses found.")
        pd.DataFrame(columns=["run_id", "n_utts", "delta_wer_obs", "delta_wer_ci_low", "delta_wer_ci_high", "p_nonpositive"]).to_csv(
            out_path, index=False
        )
        print(f"Wrote empty: {out_path}")
        return

    df = pd.DataFrame(rows).sort_values("run_id")
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(df)} runs)")


if __name__ == "__main__":
    main()
