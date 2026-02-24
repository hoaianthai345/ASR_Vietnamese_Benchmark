from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from vnasrbench.data.manifest import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    durations: List[float] = []
    missing_duration = 0
    n = 0
    for u in read_jsonl(args.manifest):
        n += 1
        if u.duration_sec and u.duration_sec > 0:
            durations.append(float(u.duration_sec))
            continue
        missing_duration += 1
        info = sf.info(u.audio_path)
        durations.append(float(info.frames) / float(info.samplerate))

    if n == 0:
        raise SystemExit("Empty manifest")

    arr = np.array(durations, dtype=np.float64)
    report: Dict[str, float] = {
        "n_utts": float(n),
        "total_sec": float(arr.sum()),
        "total_hours": float(arr.sum() / 3600.0),
        "mean_sec": float(arr.mean()),
        "p50_sec": float(np.quantile(arr, 0.50)),
        "p90_sec": float(np.quantile(arr, 0.90)),
        "p95_sec": float(np.quantile(arr, 0.95)),
        "max_sec": float(arr.max()),
        "missing_duration": float(missing_duration),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

