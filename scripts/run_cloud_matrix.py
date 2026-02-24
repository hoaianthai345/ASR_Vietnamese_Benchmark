from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from vnasrbench.cli import run_from_config


def _load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _deep_merge(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _ensure_cache_dirs(repo_root: Path) -> None:
    cache_root = repo_root / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))


def _expand_runs(runs: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for r in runs:
        datasets = r.get("datasets") or [r["dataset"]]
        models = r.get("models") or [r["model"]]
        for d in datasets:
            for m in models:
                out = dict(r)
                out["dataset"] = d
                out["model"] = m
                yield out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="configs/cloud/matrix.yaml", help="Matrix YAML")
    ap.add_argument("--filter", default="", help="Substring filter on dataset/model/run_id")
    ap.add_argument("--dry-run", action="store_true", help="Print planned runs only")
    ap.add_argument("--resume", action="store_true", help="Skip runs that already have metrics.json")
    ap.add_argument(
        "--stop-after-hours",
        type=float,
        default=0.0,
        help="Stop launching new runs after N hours elapsed (0 means no limit)",
    )
    ap.add_argument("--continue-on-error", action="store_true", help="Continue if a run fails")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_cache_dirs(repo_root)

    matrix = _load_yaml(args.matrix)
    base_cfg = _load_yaml(matrix["base_config"])
    datasets = matrix["datasets"]
    models = matrix["models"]
    runs = list(_expand_runs(matrix["runs"]))
    t_start = time.time()

    for r in runs:
        if args.stop_after_hours > 0:
            elapsed_h = (time.time() - t_start) / 3600.0
            if elapsed_h >= args.stop_after_hours:
                print(
                    f"Stop budget reached: elapsed={elapsed_h:.2f}h >= {args.stop_after_hours:.2f}h. "
                    "No more runs will be launched."
                )
                break

        dname = r["dataset"]
        mname = r["model"]
        mode = "stream" if bool(r.get("streaming", False)) else "offline"
        run_id = f"{dname}__{mname}__{mode}"

        if args.filter:
            hay = f"{dname} {mname} {run_id}"
            if args.filter not in hay:
                continue

        cfg = deepcopy(base_cfg)
        cfg["dataset"] = _load_yaml(datasets[dname])
        cfg["model"] = _load_yaml(models[mname])

        cfg.setdefault("streaming", {})
        if "streaming" in r:
            cfg["streaming"]["enabled"] = bool(r["streaming"])
        if "streaming_cfg" in r:
            _deep_merge(cfg["streaming"], r["streaming_cfg"])

        cfg.setdefault("output", {})
        run_dir = repo_root / "runs" / run_id
        metrics_path = run_dir / "metrics.json"
        if args.resume and metrics_path.exists():
            print("SKIP (resume):", run_id, "->", metrics_path)
            continue

        cfg["output"]["run_dir"] = str(run_dir)

        config_path = run_dir / "config.yaml"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

        if args.dry_run:
            print("DRY RUN:", run_id, "->", config_path)
            continue

        print("Running:", run_id)
        try:
            metrics = run_from_config(str(config_path))
            print("Done:", run_id, metrics)
        except Exception as e:
            print("FAILED:", run_id, "-", repr(e))
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
