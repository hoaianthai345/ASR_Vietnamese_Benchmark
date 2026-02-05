from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import datetime as _dt
import subprocess
import yaml

from vnasrbench.eval.runner import RunConfig, run_offline
from vnasrbench.models.hf_whisper import HFWhisperConfig, HFWhisperModel
from vnasrbench.utils.seed import set_seed
from vnasrbench.utils.io import write_json


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _git_hash(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _snapshot_run(config_path: str, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = Path(config_path).read_text(encoding="utf-8")
    (run_dir / "config.yaml").write_text(cfg_text, encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    info = {
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "git_hash": _git_hash(repo_root),
        "config_path": str(config_path),
    }
    write_json(run_dir / "run_info.json", info)


def run_from_config(path: str) -> Dict[str, Any]:
    cfg = _load_yaml(path)

    # Seed and determinism
    set_seed(int(cfg.get("seed", 42)), deterministic=True)

    run_dir = Path(cfg["output"]["run_dir"])
    _snapshot_run(path, run_dir)

    # Model
    if cfg["model"]["name"] != "hf_whisper":
        raise ValueError("Only hf_whisper is implemented in v0.1. Add adapters in src/vnasrbench/models/")

    mcfg = HFWhisperConfig(
        hf_id=cfg["model"]["hf_id"],
        language=cfg["model"].get("language", "vi"),
        task=cfg["model"].get("task", "transcribe"),
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "float16"),
        batch_size=int(cfg["model"].get("batch_size", 4)),
    )
    model = HFWhisperModel(mcfg)

    # Run
    rcfg = RunConfig(
        manifest_path=cfg["dataset"]["manifest_path"],
        run_dir=str(run_dir),
        sample_rate=int(cfg["dataset"].get("sample_rate", 16000)),
        batch_size=int(cfg["model"].get("batch_size", 4)),
        save_hyps=bool(cfg["output"].get("save_hyps", True)),
        bootstrap_samples=int(cfg.get("bootstrap_samples", 2000)),
        seed=int(cfg.get("seed", 42)),
    )

    metrics = run_offline(model, rcfg)
    return metrics


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    metrics = run_from_config(args.config)
    print("Done. Metrics:", metrics)
