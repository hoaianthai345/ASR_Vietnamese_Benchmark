from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from vnasrbench.data.manifest import Utterance, read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_in", required=True)
    ap.add_argument("--id_list", required=True, help="One utt_id per line")
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--strict", action="store_true", help="Fail if any utt_id is missing")
    ap.add_argument("--out_report", default="")
    args = ap.parse_args()

    ids: Set[str] = set()
    for line in Path(args.id_list).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.add(s)
    if not ids:
        raise SystemExit("Empty id_list")

    picked: List[Utterance] = []
    seen: Set[str] = set()
    for u in read_jsonl(args.manifest_in):
        if u.utt_id in ids:
            picked.append(u)
            seen.add(u.utt_id)

    missing = sorted(ids - seen)
    if missing and args.strict:
        raise SystemExit(f"Missing {len(missing)} utt_ids from manifest_in. First 10: {missing[:10]}")

    write_jsonl(args.out_manifest, picked)

    total_sec = sum(float(u.duration_sec) for u in picked if u.duration_sec and u.duration_sec > 0)
    report: Dict[str, float] = {
        "n_ids_requested": float(len(ids)),
        "n_utts_written": float(len(picked)),
        "n_missing_ids": float(len(missing)),
        "total_hours_known_duration": float(total_sec / 3600.0),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_report:
        Path(args.out_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

