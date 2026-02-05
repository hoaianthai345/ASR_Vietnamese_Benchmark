from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _collect_metrics(runs_dir: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["run_id"] = path.parent.name
        rows.append(metrics)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_csv", default="results.csv")
    ap.add_argument("--out_fig", default="")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = _collect_metrics(runs_dir)
    if not rows:
        raise SystemExit(f"No metrics found under {runs_dir}")

    df = pd.DataFrame(rows).sort_values("run_id")
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    if args.out_fig:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df["run_id"], df["wer"])
        ax.set_ylabel("WER")
        ax.set_xlabel("Run")
        ax.set_title("WER by Run")
        ax.set_ylim(0, max(0.1, float(df["wer"].max()) * 1.1))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(args.out_fig)
        print(f"Wrote {args.out_fig}")


if __name__ == "__main__":
    main()
