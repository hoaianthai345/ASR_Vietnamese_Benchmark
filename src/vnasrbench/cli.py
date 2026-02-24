from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import datetime as _dt
import subprocess
import platform
import sys
import yaml

from vnasrbench.data.hf import HFDatasetConfig, compute_id_list_sha256
from vnasrbench.eval.runner import RunConfig, StreamingSimConfig, run_benchmark, run_benchmark_hf
from vnasrbench.models.faster_whisper import FasterWhisperConfig, FasterWhisperModel
from vnasrbench.models.hf_ctc import HFCTCConfig, HFCTCModel
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


def _snapshot_run(
    config_path: str,
    cfg: Dict[str, Any],
    run_dir: Path,
    *,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = Path(config_path).read_text(encoding="utf-8")
    (run_dir / "config.yaml").write_text(cfg_text, encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    git_hash = _git_hash(repo_root) or "unknown"
    info = {
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "git_hash": git_hash,
        "config_path": str(config_path),
    }
    if extra_info:
        info.update(extra_info)
    write_json(run_dir / "run_info.json", info)
    (run_dir / "git_hash.txt").write_text(git_hash + "\n", encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _resolve_hf_revision(repo_id: str, repo_type: str, revision: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi()
        if repo_type == "model":
            info = api.model_info(repo_id, revision=revision)
        elif repo_type == "dataset":
            info = api.dataset_info(repo_id, revision=revision)
        else:
            return None
        return getattr(info, "sha", None)
    except Exception:
        return None


def _collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "python_executable": sys.executable,
    }
    try:
        import torch  # type: ignore

        info.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None),
                "cudnn_version": torch.backends.cudnn.version()
                if hasattr(torch.backends, "cudnn")
                else None,
                "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "gpu_names": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
                "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            }
        )
    except Exception:
        pass
    try:
        import transformers  # type: ignore

        info["transformers_version"] = transformers.__version__
    except Exception:
        pass
    try:
        import datasets  # type: ignore

        info["datasets_version"] = datasets.__version__
    except Exception:
        pass
    try:
        import faster_whisper  # type: ignore

        info["faster_whisper_version"] = faster_whisper.__version__
    except Exception:
        pass
    try:
        import ctranslate2  # type: ignore

        info["ctranslate2_version"] = ctranslate2.__version__
    except Exception:
        pass
    return info


def _validate_streaming_config(scfg: StreamingSimConfig) -> None:
    if not scfg.enabled:
        return
    if scfg.chunk_ms <= 0:
        raise ValueError("streaming.chunk_ms must be > 0")
    if scfg.overlap_ms < 0:
        raise ValueError("streaming.overlap_ms must be >= 0")
    if scfg.overlap_ms >= scfg.chunk_ms:
        raise ValueError("streaming.overlap_ms must be < chunk_ms")
    if scfg.lookahead_ms < 0:
        raise ValueError("streaming.lookahead_ms must be >= 0")


def run_from_config(path: str) -> Dict[str, Any]:
    cfg = _load_yaml(path)

    # Seed and determinism
    set_seed(int(cfg.get("seed", 42)), deterministic=True)

    run_dir = Path(cfg["output"]["run_dir"])

    # Model
    mname = cfg["model"]["name"]
    model_meta: Dict[str, Any] = {"model_name": mname}
    if mname == "hf_whisper":
        decode_cfg = cfg.get("decode", {}) or {}
        decode_cfg = {k: v for k, v in decode_cfg.items() if v is not None}
        mcfg = HFWhisperConfig(
            hf_id=cfg["model"]["hf_id"],
            hf_revision=cfg["model"].get("hf_revision"),
            language=cfg["model"].get("language", "vi"),
            task=cfg["model"].get("task", "transcribe"),
            device=cfg.get("device", "cuda"),
            dtype=cfg.get("dtype", "float16"),
            batch_size=int(cfg["model"].get("batch_size", 4)),
            decode_kwargs=decode_cfg,
        )
        model_meta.update(
            {
                "model_hf_id": mcfg.hf_id,
                "model_hf_revision": mcfg.hf_revision,
                "model_hf_resolved_sha": _resolve_hf_revision(mcfg.hf_id, "model", mcfg.hf_revision),
                "decode_kwargs": decode_cfg,
            }
        )
        model = HFWhisperModel(mcfg)
    elif mname == "hf_ctc":
        mcfg = HFCTCConfig(
            hf_id=cfg["model"]["hf_id"],
            hf_revision=cfg["model"].get("hf_revision"),
            device=cfg.get("device", "cuda"),
            dtype=cfg.get("dtype", "float16"),
            batch_size=int(cfg["model"].get("batch_size", 4)),
        )
        model_meta.update(
            {
                "model_hf_id": mcfg.hf_id,
                "model_hf_revision": mcfg.hf_revision,
                "model_hf_resolved_sha": _resolve_hf_revision(mcfg.hf_id, "model", mcfg.hf_revision),
            }
        )
        model = HFCTCModel(mcfg)
    elif mname == "faster_whisper":
        mcfg = FasterWhisperConfig(
            hf_id=cfg["model"]["hf_id"],
            hf_revision=cfg["model"].get("hf_revision"),
            device=cfg.get("device", "cpu"),
            dtype=cfg.get("dtype", "float32"),
            compute_type=cfg["model"].get("compute_type"),
            language=cfg["model"].get("language", "vi"),
            task=cfg["model"].get("task", "transcribe"),
            beam_size=int(cfg["model"].get("beam_size", 1)),
            temperature=float(cfg["model"].get("temperature", 0.0)),
            vad_filter=bool(cfg["model"].get("vad_filter", False)),
            download_root=cfg["model"].get("download_root"),
        )
        model_meta.update(
            {
                "model_hf_id": mcfg.hf_id,
                "model_hf_revision": mcfg.hf_revision,
                "model_hf_resolved_sha": _resolve_hf_revision(mcfg.hf_id, "model", mcfg.hf_revision),
                "decode_kwargs": {
                    "beam_size": mcfg.beam_size,
                    "temperature": mcfg.temperature,
                    "vad_filter": mcfg.vad_filter,
                },
            }
        )
        model = FasterWhisperModel(mcfg)
    else:
        raise ValueError(
            f"Unsupported model.name: {mname}. Supported: hf_whisper, hf_ctc, faster_whisper"
        )

    # Run
    scfg = cfg.get("streaming", {}) or {}
    streaming = StreamingSimConfig(
        enabled=bool(scfg.get("enabled", False)),
        chunk_ms=int(scfg.get("chunk_ms", 320)),
        overlap_ms=int(scfg.get("overlap_ms", 80)),
        lookahead_ms=int(scfg.get("lookahead_ms", 0)),
        stitch_tail_window_tokens=int(scfg.get("stitch_tail_window_tokens", 50)),
    )
    _validate_streaming_config(streaming)
    dcfg = cfg.get("dataset", {}) or {}
    data_source = dcfg.get("source", "manifest")
    dataset_meta: Dict[str, Any] = {
        "dataset_source": data_source,
    }
    rcfg = RunConfig(
        manifest_path=dcfg.get("manifest_path", ""),
        run_dir=str(run_dir),
        sample_rate=int(dcfg.get("sample_rate", 16000)),
        batch_size=int(cfg["model"].get("batch_size", 4)),
        save_hyps=bool(cfg["output"].get("save_hyps", True)),
        save_streaming_partials=bool(cfg["output"].get("save_streaming_partials", False)),
        bootstrap_samples=int(cfg.get("bootstrap_samples", 2000)),
        seed=int(cfg.get("seed", 42)),
        streaming=streaming,
    )

    if data_source == "hf":
        id_list_path = dcfg.get("id_list_path")
        dataset_meta.update(
            {
                "dataset_hf_id": dcfg.get("hf_id"),
                "dataset_hf_split": dcfg.get("hf_split", "train"),
                "dataset_text_key": dcfg.get("text_key"),
                "dataset_id_key": dcfg.get("id_key"),
                "dataset_hf_revision": dcfg.get("hf_revision"),
                "dataset_hf_resolved_sha": _resolve_hf_revision(
                    str(dcfg.get("hf_id")), "dataset", dcfg.get("hf_revision")
                ),
                "dataset_id_list_path": id_list_path,
                "dataset_id_list_sha256": compute_id_list_sha256(id_list_path),
                "dataset_limit": dcfg.get("limit"),
                "dataset_streaming": bool(dcfg.get("streaming", True)),
            }
        )

        warnings: list[str] = []
        if str(dcfg.get("hf_split", "train")).lower() == "train" and not id_list_path:
            warnings.append("hf_split=train without id_list_path; evaluation may not be held-out.")
        if warnings:
            dataset_meta["config_warnings"] = warnings
            for w in warnings:
                print(f"[WARN] {w}")

        env_info = _collect_env_info()
        _snapshot_run(path, cfg, run_dir, extra_info={**env_info, **dataset_meta, **model_meta})

        hf_cfg = HFDatasetConfig(
            hf_id=str(dcfg["hf_id"]),
            split=str(dcfg.get("hf_split", "train")),
            text_key=str(dcfg["text_key"]),
            id_key=dcfg.get("id_key"),
            streaming=bool(dcfg.get("streaming", True)),
            limit=dcfg.get("limit"),
            id_list_path=dcfg.get("id_list_path"),
            revision=dcfg.get("hf_revision"),
        )
        metrics = run_benchmark_hf(model, rcfg, hf_cfg)
    else:
        env_info = _collect_env_info()
        dataset_meta.update(
            {
                "dataset_manifest_path": dcfg.get("manifest_path"),
                "dataset_sample_rate": dcfg.get("sample_rate", 16000),
            }
        )
        _snapshot_run(path, cfg, run_dir, extra_info={**env_info, **dataset_meta, **model_meta})
        if not rcfg.manifest_path:
            raise ValueError("dataset.manifest_path is required for source=manifest")
        metrics = run_benchmark(model, rcfg)
    return metrics


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    metrics = run_from_config(args.config)
    print("Done. Metrics:", metrics)
