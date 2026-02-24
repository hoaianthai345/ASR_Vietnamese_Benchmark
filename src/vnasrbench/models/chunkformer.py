from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

from vnasrbench.models.base import ASRModel, DecodeResult


@dataclass
class ChunkFormerConfig:
    hf_id: str
    hf_revision: Optional[str] = None
    device: str = "cuda"
    chunk_size: int = 64
    left_context_size: int = 128
    right_context_size: int = 128
    total_batch_duration: int = 1800


class ChunkFormerModel(ASRModel):
    def __init__(self, cfg: ChunkFormerConfig) -> None:
        self.cfg = cfg
        try:
            from chunkformer import ChunkFormerModel as _ChunkFormer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: chunkformer. Install with: pip install -U chunkformer"
            ) from e

        kwargs = {}
        if cfg.hf_revision:
            kwargs["revision"] = cfg.hf_revision

        self.model = _ChunkFormer.from_pretrained(cfg.hf_id, **kwargs).to(cfg.device)
        self.model.eval()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="vnasrbench_chunkformer_")
        self._min_window_ms = 25.0
        try:
            fbank_conf = getattr(self.model.config, "fbank_conf", None)
            if isinstance(fbank_conf, dict):
                self._min_window_ms = float(fbank_conf.get("frame_length", self._min_window_ms))
        except Exception:
            pass

    def _prepare_wav(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        out = np.asarray(wav, dtype=np.float32)
        if out.ndim > 1:
            out = out.mean(axis=-1)
        window_samples = int(round(sample_rate * (self._min_window_ms / 1000.0)))
        # pydub/ffmpeg can shave a few samples on very short waveforms,
        # so keep a safety margin above the theoretical minimum window size.
        min_samples = max(2, window_samples + max(32, window_samples // 4))
        if out.shape[0] < min_samples:
            out = np.pad(out, (0, min_samples - out.shape[0]), mode="constant")
        return out

    def _decode_to_text(self, x: object) -> str:
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, dict):
            # Defensive fallback for variants that may return structured output.
            if "decode" in x:
                return str(x["decode"]).strip()
            return str(x).strip()
        if isinstance(x, (list, tuple)):
            return " ".join(str(t) for t in x).strip()
        return str(x).strip()

    def transcribe_batch(self, wavs: List[np.ndarray], sample_rate: int) -> List[DecodeResult]:
        t0 = time.perf_counter()
        temp_paths: List[Path] = []
        try:
            for i, wav in enumerate(wavs):
                wav = self._prepare_wav(wav, sample_rate)
                wav_path = Path(self._tmpdir.name) / f"chunkformer_{os.getpid()}_{i}.wav"
                sf.write(str(wav_path), wav, sample_rate)
                temp_paths.append(wav_path)

            decodes = self.model.batch_decode(
                [str(p) for p in temp_paths],
                chunk_size=self.cfg.chunk_size,
                left_context_size=self.cfg.left_context_size,
                right_context_size=self.cfg.right_context_size,
                total_batch_duration=self.cfg.total_batch_duration,
            )
            texts = [self._decode_to_text(d) for d in decodes]
        finally:
            for p in temp_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        if len(texts) != len(wavs):
            raise RuntimeError(
                f"ChunkFormer returned {len(texts)} outputs for {len(wavs)} inputs"
            )

        t1 = time.perf_counter()
        audio_sec = sum(len(w) for w in wavs) / float(sample_rate)
        compute_sec = max(1e-9, (t1 - t0))
        rtf = compute_sec / max(1e-9, audio_sec)
        return [DecodeResult(text=t, rtf=rtf) for t in texts]
