from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time

import numpy as np
import soundfile as sf

from vnasrbench.data.manifest import read_jsonl
from vnasrbench.eval.wer import wer, paired_samples
from vnasrbench.utils.bootstrap import bootstrap_wer_ci
from vnasrbench.utils.textnorm import normalize_vi
from vnasrbench.models.base import ASRModel


@dataclass(frozen=True)
class RunConfig:
    manifest_path: str
    run_dir: str
    sample_rate: int = 16000
    batch_size: int = 4
    save_hyps: bool = True
    bootstrap_samples: int = 2000
    seed: int = 42


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        # minimal resample without extra deps is non-trivial; enforce sr for now
        # For best paper: implement resample via torchaudio or librosa and document it.
        raise ValueError(f"Sample rate mismatch: got {sr}, expected {target_sr}. Pre-resample your data.")
    return wav


def run_offline(model: ASRModel, cfg: RunConfig) -> Dict[str, float]:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    refs: List[str] = []
    hyps: List[str] = []
    rtfs: List[float] = []

    t_start = time.perf_counter()
    batch_wavs: List[np.ndarray] = []
    batch_utts: List[Tuple[str, str]] = []

    for u in read_jsonl(cfg.manifest_path):
        wav = _load_audio(u.audio_path, cfg.sample_rate)
        batch_wavs.append(wav)
        batch_utts.append((u.utt_id, u.text))

        if len(batch_wavs) == cfg.batch_size:
            outs = model.transcribe_batch(batch_wavs, cfg.sample_rate)
            for (_utt_id, ref), out in zip(batch_utts, outs):
                refs.append(ref)
                hyps.append(out.text)
                rtfs.append(out.rtf)
            batch_wavs, batch_utts = [], []

    if batch_wavs:
        outs = model.transcribe_batch(batch_wavs, cfg.sample_rate)
        for (_utt_id, ref), out in zip(batch_utts, outs):
            refs.append(ref)
            hyps.append(out.text)
            rtfs.append(out.rtf)

    t_end = time.perf_counter()

    wr = wer(refs, hyps)
    ci = bootstrap_wer_ci(paired_samples(refs, hyps), n_samples=cfg.bootstrap_samples, seed=cfg.seed)
    avg_rtf = float(np.mean(rtfs)) if rtfs else -1.0

    metrics = {
        "wer": float(wr.wer),
        "wer_ci_mean": float(ci.mean),
        "wer_ci_low": float(ci.low),
        "wer_ci_high": float(ci.high),
        "avg_rtf": avg_rtf,
        "wall_time_sec": float(t_end - t_start),
        "n_utts": float(len(refs)),
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.save_hyps:
        out_path = run_dir / "hypotheses.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for i, (r, h) in enumerate(zip(refs, hyps)):
                f.write(
                    json.dumps(
                        {
                            "idx": i,
                            "ref": r,
                            "hyp": h,
                            "ref_norm": normalize_vi(r),
                            "hyp_norm": normalize_vi(h),
                        },
                        ensure_ascii=False,
                    )
                    + "
"
                )

    return metrics
