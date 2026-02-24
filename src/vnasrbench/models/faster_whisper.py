from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from vnasrbench.models.base import ASRModel, DecodeResult


@dataclass
class FasterWhisperConfig:
    hf_id: str
    hf_revision: Optional[str] = None
    device: str = "cpu"  # "cpu" | "cuda" | "auto"
    dtype: str = "float32"  # maps to compute_type if compute_type is not set
    compute_type: Optional[str] = None  # e.g. "float16", "int8"
    language: str = "vi"
    task: str = "transcribe"
    beam_size: int = 1
    temperature: float = 0.0
    vad_filter: bool = False
    download_root: Optional[str] = None


class FasterWhisperModel(ASRModel):
    def __init__(self, cfg: FasterWhisperConfig) -> None:
        self.cfg = cfg
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: faster-whisper. Install with: pip install -U faster-whisper"
            ) from e

        compute_type = cfg.compute_type
        if compute_type is None:
            compute_type = "float16" if cfg.dtype == "float16" else "float32"

        model_id = cfg.hf_id
        if cfg.hf_revision:
            try:
                from huggingface_hub import snapshot_download  # type: ignore

                model_id = snapshot_download(
                    cfg.hf_id,
                    revision=cfg.hf_revision,
                    repo_type="model",
                    local_dir=None,
                )
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Failed to resolve faster-whisper revision. "
                    "Install/upgrade huggingface_hub or remove hf_revision."
                ) from e

        self.model = WhisperModel(
            model_id,
            device=cfg.device,
            compute_type=compute_type,
            download_root=cfg.download_root,
        )

    def transcribe_batch(self, wavs: List[np.ndarray], sample_rate: int) -> List[DecodeResult]:
        outs: List[DecodeResult] = []
        for wav in wavs:
            t0 = time.perf_counter()
            segments, _info = self.model.transcribe(
                wav,
                language=self.cfg.language,
                task=self.cfg.task,
                beam_size=self.cfg.beam_size,
                temperature=self.cfg.temperature,
                vad_filter=self.cfg.vad_filter,
            )
            text = "".join(seg.text for seg in segments).strip()
            t1 = time.perf_counter()

            audio_sec = float(len(wav)) / float(sample_rate)
            compute_sec = max(1e-9, (t1 - t0))
            rtf = compute_sec / max(1e-9, audio_sec)

            outs.append(DecodeResult(text=text, rtf=rtf))
        return outs
