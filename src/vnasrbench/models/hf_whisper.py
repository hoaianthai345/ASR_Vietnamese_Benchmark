from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from vnasrbench.models.base import ASRModel, DecodeResult


@dataclass
class HFWhisperConfig:
    hf_id: str
    hf_revision: Optional[str] = None
    language: str = "vi"
    task: str = "transcribe"
    device: str = "cuda"
    dtype: str = "float16"
    batch_size: int = 4
    decode_kwargs: Dict[str, Any] = field(default_factory=dict)


class HFWhisperModel(ASRModel):
    def __init__(self, cfg: HFWhisperConfig) -> None:
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.hf_id)
        self.torch_dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.hf_id,
            torch_dtype=self.torch_dtype,
            revision=cfg.hf_revision,
        )
        self.model.to(cfg.device, dtype=self.torch_dtype)
        self.model.eval()
        # Prefer language/task over forced_decoder_ids to avoid deprecation warnings.
        try:
            if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                self.model.generation_config.language = cfg.language
                self.model.generation_config.task = cfg.task
                self.model.generation_config.forced_decoder_ids = None
        except Exception:
            pass

    @torch.inference_mode()
    def transcribe_batch(self, wavs: List[np.ndarray], sample_rate: int) -> List[DecodeResult]:
        t0 = time.perf_counter()
        inputs = self.processor(
            wavs,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.cfg.device, dtype=self.torch_dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.cfg.device)

        gen_kwargs = dict(self.cfg.decode_kwargs)
        # Fallback to forced_decoder_ids if generation_config doesn't support language/task.
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is None or not hasattr(gen_cfg, "language") or not hasattr(gen_cfg, "task"):
            gen_kwargs["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                language=self.cfg.language, task=self.cfg.task
            )
        gen = self.model.generate(
            input_features,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        texts = self.processor.batch_decode(gen, skip_special_tokens=True)
        t1 = time.perf_counter()

        # RTF: compute_time / total_audio_time
        audio_sec = sum(len(w) for w in wavs) / float(sample_rate)
        compute_sec = max(1e-9, (t1 - t0))
        rtf = compute_sec / max(1e-9, audio_sec)

        return [DecodeResult(text=t.strip(), rtf=rtf) for t in texts]
