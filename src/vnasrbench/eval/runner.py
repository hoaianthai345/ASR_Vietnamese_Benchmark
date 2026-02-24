from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import time
import hashlib

import numpy as np
import soundfile as sf

from vnasrbench.data.hf import HFDatasetConfig, iter_hf_dataset
from vnasrbench.data.manifest import read_jsonl
from vnasrbench.eval.wer import paired_samples, wer, cer
from vnasrbench.models.base import ASRModel
from vnasrbench.streaming.chunker import ChunkConfig, iter_chunks
from vnasrbench.streaming.stitch import compute_stability, lcs_merge_tokens
from vnasrbench.utils.bootstrap import bootstrap_wer_ci
from vnasrbench.utils.textnorm import mask_numbers, normalize_wer_strict_vi, normalize_cer_strict_vi


@dataclass(frozen=True)
class StreamingSimConfig:
    enabled: bool = False
    chunk_ms: int = 320
    overlap_ms: int = 80
    lookahead_ms: int = 0
    stitch_tail_window_tokens: int = 50

    def latency_proxy_ms(self) -> int:
        # Paper definition: algorithmic latency proxy = chunk + lookahead (no I/O).
        return int(self.chunk_ms + self.lookahead_ms)

    def to_chunk_config(self, sample_rate: int) -> ChunkConfig:
        return ChunkConfig(
            sample_rate=sample_rate,
            chunk_ms=self.chunk_ms,
            overlap_ms=self.overlap_ms,
            lookahead_ms=self.lookahead_ms,
        )


@dataclass(frozen=True)
class RunConfig:
    manifest_path: str
    run_dir: str
    sample_rate: int = 16000
    batch_size: int = 4
    save_hyps: bool = True
    save_streaming_partials: bool = False
    bootstrap_samples: int = 2000
    seed: int = 42
    streaming: StreamingSimConfig = StreamingSimConfig()


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        raise ValueError(
            f"Sample rate mismatch: got {sr}, expected {target_sr}. Pre-resample your data."
        )
    return wav


def _resample_if_needed(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    try:
        import torch
        import torchaudio

        wav_t = torch.from_numpy(wav).float().unsqueeze(0)
        resampled = torchaudio.functional.resample(wav_t, sr, target_sr)
        return resampled.squeeze(0).cpu().numpy()
    except Exception as e:
        raise ValueError(
            f"Sample rate mismatch: got {sr}, expected {target_sr}. "
            "Install torchaudio to enable resampling."
        ) from e


def _wer_num_masked(refs: Sequence[str], hyps: Sequence[str]) -> float:
    def norm(s: str) -> str:
        return normalize_wer_strict_vi(mask_numbers(s))

    return float(wer(refs, hyps, normalizer=norm).wer)


def _cer_strict(refs: Sequence[str], hyps: Sequence[str]) -> float:
    def norm(s: str) -> str:
        return normalize_cer_strict_vi(s)

    return float(cer(refs, hyps, normalizer=norm).cer)


def _score_bundle(
    refs: List[str],
    hyps: List[str],
    *,
    bootstrap_samples: int,
    seed: int,
    prefix: str,
) -> Dict[str, Any]:
    wr = wer(refs, hyps, normalizer=normalize_wer_strict_vi)
    ci = bootstrap_wer_ci(
        paired_samples(refs, hyps),
        n_samples=bootstrap_samples,
        seed=seed,
        score_fn=lambda r, h: wer(r, h, normalizer=normalize_wer_strict_vi).wer,
    )

    num_ci = bootstrap_wer_ci(
        paired_samples(refs, hyps),
        n_samples=bootstrap_samples,
        seed=seed,
        score_fn=_wer_num_masked,
    )

    cer_ci = bootstrap_wer_ci(
        paired_samples(refs, hyps),
        n_samples=bootstrap_samples,
        seed=seed,
        score_fn=_cer_strict,
    )
    cer_val = cer(refs, hyps, normalizer=normalize_cer_strict_vi)

    return {
        f"{prefix}_wer": float(wr.wer),
        f"{prefix}_wer_ci_mean": float(ci.mean),
        f"{prefix}_wer_ci_low": float(ci.low),
        f"{prefix}_wer_ci_high": float(ci.high),
        f"{prefix}_wer_num_masked": float(_wer_num_masked(refs, hyps)),
        f"{prefix}_wer_num_masked_ci_mean": float(num_ci.mean),
        f"{prefix}_wer_num_masked_ci_low": float(num_ci.low),
        f"{prefix}_wer_num_masked_ci_high": float(num_ci.high),
        f"{prefix}_n_ref_words": float(wr.n_ref_words),
        f"{prefix}_cer": float(cer_val.cer),
        f"{prefix}_cer_ci_mean": float(cer_ci.mean),
        f"{prefix}_cer_ci_low": float(cer_ci.low),
        f"{prefix}_cer_ci_high": float(cer_ci.high),
        f"{prefix}_n_ref_chars": float(cer_val.n_ref_chars),
        f"{prefix}_wer_norm": "strict_vi",
        f"{prefix}_wer_num_masked_norm": "strict_vi+mask_numbers",
        f"{prefix}_cer_norm": "strict_vi_no_space",
    }


def _update_order_hash(h: "hashlib._Hash", utt_id: str) -> None:
    h.update((utt_id + "\n").encode("utf-8"))


def _decode_offline(
    model: ASRModel, cfg: RunConfig
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    refs: List[str] = []
    hyps: List[str] = []
    hyp_by_id: Dict[str, str] = {}
    ref_by_id: Dict[str, str] = {}

    total_audio_sec = 0.0
    total_infer_sec = 0.0
    order_hash = hashlib.sha256()

    batch_wavs: List[np.ndarray] = []
    batch_ids: List[str] = []
    batch_refs: List[str] = []

    for u in read_jsonl(cfg.manifest_path):
        wav = _load_audio(u.audio_path, cfg.sample_rate)
        batch_wavs.append(wav)
        batch_ids.append(u.utt_id)
        batch_refs.append(u.text)
        total_audio_sec += float(len(wav)) / float(cfg.sample_rate)
        _update_order_hash(order_hash, u.utt_id)

        if len(batch_wavs) == cfg.batch_size:
            t0 = time.perf_counter()
            outs = model.transcribe_batch(batch_wavs, cfg.sample_rate)
            t1 = time.perf_counter()
            total_infer_sec += float(t1 - t0)
            for utt_id, ref, out in zip(batch_ids, batch_refs, outs):
                refs.append(ref)
                hyps.append(out.text)
                hyp_by_id[utt_id] = out.text
                ref_by_id[utt_id] = ref
            batch_wavs, batch_ids, batch_refs = [], [], []

    if batch_wavs:
        t0 = time.perf_counter()
        outs = model.transcribe_batch(batch_wavs, cfg.sample_rate)
        t1 = time.perf_counter()
        total_infer_sec += float(t1 - t0)
        for utt_id, ref, out in zip(batch_ids, batch_refs, outs):
            refs.append(ref)
            hyps.append(out.text)
            hyp_by_id[utt_id] = out.text
            ref_by_id[utt_id] = ref

    rtf = float(total_infer_sec) / float(max(1e-9, total_audio_sec))
    metrics = _score_bundle(refs, hyps, bootstrap_samples=cfg.bootstrap_samples, seed=cfg.seed, prefix="offline")
    metrics.update(
        {
            "offline_rtf": rtf,
            "offline_total_audio_sec": float(total_audio_sec),
            "offline_total_infer_sec": float(total_infer_sec),
            "n_utts": float(len(refs)),
            "offline_order_hash": order_hash.hexdigest(),
        }
    )
    return metrics, hyp_by_id, ref_by_id


def _decode_streaming_sim(
    model: ASRModel,
    cfg: RunConfig,
    *,
    ref_by_id: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, List[str]]]:
    if not cfg.streaming.enabled:
        raise ValueError("streaming is disabled in config")

    refs: List[str] = []
    hyps: List[str] = []
    hyp_by_id: Dict[str, str] = {}
    partials_by_id: Dict[str, List[str]] = {}

    total_audio_sec = 0.0
    total_infer_sec = 0.0
    n_chunks_total = 0
    order_hash = hashlib.sha256()

    ccfg = cfg.streaming.to_chunk_config(cfg.sample_rate)

    for u in read_jsonl(cfg.manifest_path):
        wav = _load_audio(u.audio_path, cfg.sample_rate)
        total_audio_sec += float(len(wav)) / float(cfg.sample_rate)
        _update_order_hash(order_hash, u.utt_id)

        merged_tokens: List[str] = []
        partials: List[str] = []

        for ch in iter_chunks(wav, ccfg):
            t0 = time.perf_counter()
            out = model.transcribe_batch([ch], cfg.sample_rate)[0]
            t1 = time.perf_counter()
            total_infer_sec += float(t1 - t0)

            cur_tokens = normalize_wer_strict_vi(out.text).split()
            merged_tokens = lcs_merge_tokens(
                merged_tokens,
                cur_tokens,
                tail_window_tokens=cfg.streaming.stitch_tail_window_tokens,
            )
            partials.append(" ".join(merged_tokens))
            n_chunks_total += 1

        hyp = " ".join(merged_tokens).strip()
        hyp_by_id[u.utt_id] = hyp
        partials_by_id[u.utt_id] = partials

        ref = ref_by_id[u.utt_id] if ref_by_id is not None and u.utt_id in ref_by_id else u.text
        refs.append(ref)
        hyps.append(hyp)

    rtf = float(total_infer_sec) / float(max(1e-9, total_audio_sec))
    metrics = _score_bundle(
        refs, hyps, bootstrap_samples=cfg.bootstrap_samples, seed=cfg.seed, prefix="streaming"
    )

    # Stability metrics (appendix): computed on merged partials.
    st_items = [compute_stability(partials) for partials in partials_by_id.values() if partials]
    if st_items:
        st_edit_overhead = float(np.mean([s.edit_overhead for s in st_items]))
        st_change_rate = float(np.mean([s.change_rate for s in st_items]))
        st_avg_step_edits = float(np.mean([s.avg_step_edits for s in st_items]))
    else:
        st_edit_overhead = 0.0
        st_change_rate = 0.0
        st_avg_step_edits = 0.0

    metrics.update(
        {
            "streaming_rtf": rtf,
            "streaming_total_audio_sec": float(total_audio_sec),
            "streaming_total_infer_sec": float(total_infer_sec),
            "streaming_latency_proxy_ms": float(cfg.streaming.latency_proxy_ms()),
            "streaming_chunk_ms": float(cfg.streaming.chunk_ms),
            "streaming_overlap_ms": float(cfg.streaming.overlap_ms),
            "streaming_lookahead_ms": float(cfg.streaming.lookahead_ms),
            "streaming_n_chunks_total": float(n_chunks_total),
            "streaming_avg_chunks_per_utt": float(n_chunks_total) / float(max(1, len(refs))),
            "stability_edit_overhead": st_edit_overhead,
            "stability_change_rate": st_change_rate,
            "stability_avg_step_edits": st_avg_step_edits,
            "streaming_mode": "simulation",
            "stitch_policy": "lcs",
            "stateful": False,
            "streaming_order_hash": order_hash.hexdigest(),
        }
    )
    return metrics, hyp_by_id, partials_by_id


def _decode_offline_hf(
    model: ASRModel,
    cfg: RunConfig,
    hf_cfg: HFDatasetConfig,
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    refs: List[str] = []
    hyps: List[str] = []
    hyp_by_id: Dict[str, str] = {}
    ref_by_id: Dict[str, str] = {}

    total_audio_sec = 0.0
    total_infer_sec = 0.0
    order_hash = hashlib.sha256()

    batch_wavs: List[np.ndarray] = []
    batch_ids: List[str] = []
    batch_refs: List[str] = []

    for utt_id, text, wav, sr in iter_hf_dataset(hf_cfg):
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        wav = _resample_if_needed(wav, sr, cfg.sample_rate)
        total_audio_sec += float(len(wav)) / float(cfg.sample_rate)
        _update_order_hash(order_hash, utt_id)

        batch_wavs.append(wav)
        batch_ids.append(utt_id)
        batch_refs.append(text)

        if len(batch_wavs) == cfg.batch_size:
            t0 = time.perf_counter()
            outs = model.transcribe_batch(batch_wavs, cfg.sample_rate)
            t1 = time.perf_counter()
            total_infer_sec += float(t1 - t0)
            for uid, ref, out in zip(batch_ids, batch_refs, outs):
                refs.append(ref)
                hyps.append(out.text)
                hyp_by_id[uid] = out.text
                ref_by_id[uid] = ref
            batch_wavs, batch_ids, batch_refs = [], [], []

    if batch_wavs:
        t0 = time.perf_counter()
        outs = model.transcribe_batch(batch_wavs, cfg.sample_rate)
        t1 = time.perf_counter()
        total_infer_sec += float(t1 - t0)
        for uid, ref, out in zip(batch_ids, batch_refs, outs):
            refs.append(ref)
            hyps.append(out.text)
            hyp_by_id[uid] = out.text
            ref_by_id[uid] = ref

    rtf = float(total_infer_sec) / float(max(1e-9, total_audio_sec))
    metrics = _score_bundle(refs, hyps, bootstrap_samples=cfg.bootstrap_samples, seed=cfg.seed, prefix="offline")
    metrics.update(
        {
            "offline_rtf": rtf,
            "offline_total_audio_sec": float(total_audio_sec),
            "offline_total_infer_sec": float(total_infer_sec),
            "n_utts": float(len(refs)),
            "offline_order_hash": order_hash.hexdigest(),
        }
    )
    return metrics, hyp_by_id, ref_by_id


def _decode_streaming_sim_hf(
    model: ASRModel,
    cfg: RunConfig,
    hf_cfg: HFDatasetConfig,
    *,
    ref_by_id: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, List[str]]]:
    if not cfg.streaming.enabled:
        raise ValueError("streaming is disabled in config")

    refs: List[str] = []
    hyps: List[str] = []
    hyp_by_id: Dict[str, str] = {}
    partials_by_id: Dict[str, List[str]] = {}

    total_audio_sec = 0.0
    total_infer_sec = 0.0
    n_chunks_total = 0
    order_hash = hashlib.sha256()

    ccfg = cfg.streaming.to_chunk_config(cfg.sample_rate)

    for utt_id, text, wav, sr in iter_hf_dataset(hf_cfg):
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        wav = _resample_if_needed(wav, sr, cfg.sample_rate)
        total_audio_sec += float(len(wav)) / float(cfg.sample_rate)
        _update_order_hash(order_hash, utt_id)

        merged_tokens: List[str] = []
        partials: List[str] = []

        for ch in iter_chunks(wav, ccfg):
            t0 = time.perf_counter()
            out = model.transcribe_batch([ch], cfg.sample_rate)[0]
            t1 = time.perf_counter()
            total_infer_sec += float(t1 - t0)

            cur_tokens = normalize_wer_strict_vi(out.text).split()
            merged_tokens = lcs_merge_tokens(
                merged_tokens,
                cur_tokens,
                tail_window_tokens=cfg.streaming.stitch_tail_window_tokens,
            )
            partials.append(" ".join(merged_tokens))
            n_chunks_total += 1

        hyp = " ".join(merged_tokens).strip()
        hyp_by_id[utt_id] = hyp
        partials_by_id[utt_id] = partials

        ref = ref_by_id[utt_id] if ref_by_id is not None and utt_id in ref_by_id else text
        refs.append(ref)
        hyps.append(hyp)

    rtf = float(total_infer_sec) / float(max(1e-9, total_audio_sec))
    metrics = _score_bundle(
        refs, hyps, bootstrap_samples=cfg.bootstrap_samples, seed=cfg.seed, prefix="streaming"
    )

    st_items = [compute_stability(partials) for partials in partials_by_id.values() if partials]
    if st_items:
        st_edit_overhead = float(np.mean([s.edit_overhead for s in st_items]))
        st_change_rate = float(np.mean([s.change_rate for s in st_items]))
        st_avg_step_edits = float(np.mean([s.avg_step_edits for s in st_items]))
    else:
        st_edit_overhead = 0.0
        st_change_rate = 0.0
        st_avg_step_edits = 0.0

    metrics.update(
        {
            "streaming_rtf": rtf,
            "streaming_total_audio_sec": float(total_audio_sec),
            "streaming_total_infer_sec": float(total_infer_sec),
            "streaming_latency_proxy_ms": float(cfg.streaming.latency_proxy_ms()),
            "streaming_chunk_ms": float(cfg.streaming.chunk_ms),
            "streaming_overlap_ms": float(cfg.streaming.overlap_ms),
            "streaming_lookahead_ms": float(cfg.streaming.lookahead_ms),
            "streaming_n_chunks_total": float(n_chunks_total),
            "streaming_avg_chunks_per_utt": float(n_chunks_total) / float(max(1, len(refs))),
            "stability_edit_overhead": st_edit_overhead,
            "stability_change_rate": st_change_rate,
            "stability_avg_step_edits": st_avg_step_edits,
            "streaming_mode": "simulation",
            "stitch_policy": "lcs",
            "stateful": False,
            "streaming_order_hash": order_hash.hexdigest(),
        }
    )
    return metrics, hyp_by_id, partials_by_id


def run_benchmark(model: ASRModel, cfg: RunConfig) -> Dict[str, Any]:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    offline_metrics, offline_hyps, ref_by_id = _decode_offline(model, cfg)

    streaming_metrics: Dict[str, Any] = {}
    streaming_hyps: Dict[str, str] = {}
    partials_by_id: Dict[str, List[str]] = {}
    if cfg.streaming.enabled:
        streaming_metrics, streaming_hyps, partials_by_id = _decode_streaming_sim(
            model, cfg, ref_by_id=ref_by_id
        )

        streaming_metrics["delta_wer"] = float(
            streaming_metrics["streaming_wer"] - offline_metrics["offline_wer"]
        )
        streaming_metrics["delta_wer_num_masked"] = float(
            streaming_metrics["streaming_wer_num_masked"] - offline_metrics["offline_wer_num_masked"]
        )
        if offline_metrics.get("offline_order_hash") and streaming_metrics.get("streaming_order_hash"):
            streaming_metrics["order_hash_match"] = float(
                offline_metrics["offline_order_hash"] == streaming_metrics["streaming_order_hash"]
            )

    t_end = time.perf_counter()

    metrics: Dict[str, Any] = {
        "wall_time_sec": float(t_end - t_start),
        "mode": float(1.0 if cfg.streaming.enabled else 0.0),  # 1=offline+streaming, 0=offline only
    }
    metrics.update(offline_metrics)
    metrics.update(streaming_metrics)
    # Convenience fields for aggregation scripts (single scalar WER/RTF per run).
    if cfg.streaming.enabled:
        metrics["wer"] = float(metrics["streaming_wer"])
        metrics["rtf"] = float(metrics["streaming_rtf"])
        metrics["latency_proxy_ms"] = float(metrics["streaming_latency_proxy_ms"])
    else:
        metrics["wer"] = float(metrics["offline_wer"])
        metrics["rtf"] = float(metrics["offline_rtf"])
        metrics["latency_proxy_ms"] = -1.0

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.save_hyps:
        out_path = run_dir / "hypotheses.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for utt_id, ref in ref_by_id.items():
                row: Dict[str, object] = {
                    "utt_id": utt_id,
                    "ref": ref,
                    "hyp_offline": offline_hyps.get(utt_id, ""),
                    "ref_norm_strict": normalize_wer_strict_vi(ref),
                    "hyp_offline_norm_strict": normalize_wer_strict_vi(offline_hyps.get(utt_id, "")),
                }
                if cfg.streaming.enabled:
                    sh = streaming_hyps.get(utt_id, "")
                    row.update(
                        {
                            "hyp_streaming": sh,
                            "hyp_streaming_norm_strict": normalize_wer_strict_vi(sh),
                        }
                    )
                    if cfg.save_streaming_partials:
                        row["partials_streaming"] = partials_by_id.get(utt_id, [])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


def run_benchmark_hf(model: ASRModel, cfg: RunConfig, hf_cfg: HFDatasetConfig) -> Dict[str, Any]:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    offline_metrics, offline_hyps, ref_by_id = _decode_offline_hf(model, cfg, hf_cfg)

    streaming_metrics: Dict[str, Any] = {}
    streaming_hyps: Dict[str, str] = {}
    partials_by_id: Dict[str, List[str]] = {}
    if cfg.streaming.enabled:
        streaming_metrics, streaming_hyps, partials_by_id = _decode_streaming_sim_hf(
            model, cfg, hf_cfg, ref_by_id=ref_by_id
        )

        streaming_metrics["delta_wer"] = float(
            streaming_metrics["streaming_wer"] - offline_metrics["offline_wer"]
        )
        streaming_metrics["delta_wer_num_masked"] = float(
            streaming_metrics["streaming_wer_num_masked"] - offline_metrics["offline_wer_num_masked"]
        )
        if offline_metrics.get("offline_order_hash") and streaming_metrics.get("streaming_order_hash"):
            streaming_metrics["order_hash_match"] = float(
                offline_metrics["offline_order_hash"] == streaming_metrics["streaming_order_hash"]
            )

    t_end = time.perf_counter()

    metrics: Dict[str, Any] = {
        "wall_time_sec": float(t_end - t_start),
        "mode": float(1.0 if cfg.streaming.enabled else 0.0),
    }
    metrics.update(offline_metrics)
    metrics.update(streaming_metrics)

    if cfg.streaming.enabled:
        metrics["wer"] = float(metrics["streaming_wer"])
        metrics["rtf"] = float(metrics["streaming_rtf"])
        metrics["latency_proxy_ms"] = float(metrics["streaming_latency_proxy_ms"])
    else:
        metrics["wer"] = float(metrics["offline_wer"])
        metrics["rtf"] = float(metrics["offline_rtf"])
        metrics["latency_proxy_ms"] = -1.0

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.save_hyps:
        out_path = run_dir / "hypotheses.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for utt_id, ref in ref_by_id.items():
                row: Dict[str, object] = {
                    "utt_id": utt_id,
                    "ref": ref,
                    "hyp_offline": offline_hyps.get(utt_id, ""),
                    "ref_norm_strict": normalize_wer_strict_vi(ref),
                    "hyp_offline_norm_strict": normalize_wer_strict_vi(offline_hyps.get(utt_id, "")),
                }
                if cfg.streaming.enabled:
                    sh = streaming_hyps.get(utt_id, "")
                    row.update(
                        {
                            "hyp_streaming": sh,
                            "hyp_streaming_norm_strict": normalize_wer_strict_vi(sh),
                        }
                    )
                    if cfg.save_streaming_partials:
                        row["partials_streaming"] = partials_by_id.get(utt_id, [])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics
