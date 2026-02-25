from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

from vnasrbench.models.base import ASRModel, DecodeResult


@dataclass
class HFCTCConfig:
    hf_id: str
    hf_revision: Optional[str] = None
    device: str = "cuda"
    dtype: str = "float16"
    batch_size: int = 4


class HFCTCModel(ASRModel):
    def __init__(self, cfg: HFCTCConfig) -> None:
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.hf_id)
        self.torch_dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        self.model = AutoModelForCTC.from_pretrained(
            cfg.hf_id,
            dtype=self.torch_dtype,
            revision=cfg.hf_revision,
        )
        self.model.to(cfg.device, dtype=self.torch_dtype)
        self.model.eval()
        self.min_input_samples = self._infer_min_input_samples(default=400)

    def _infer_min_input_samples(self, default: int = 400) -> int:
        """
        Compute minimal raw waveform length so all conv feature-extractor layers are valid.
        For wav2vec2-like stacks this is typically 400 samples.
        """
        try:
            kernels = list(getattr(self.model.config, "conv_kernel", []) or [])
            strides = list(getattr(self.model.config, "conv_stride", []) or [])
            if not kernels or not strides or len(kernels) != len(strides):
                return default

            need = int(kernels[-1])
            for k, s in zip(reversed(kernels[:-1]), reversed(strides[:-1])):
                need = int((need - 1) * int(s) + int(k))
            # Add a tiny safety margin for edge rounding in downstream ops.
            return max(default, int(need) + 8)
        except Exception:
            return default

    def _prepare_wav(self, wav: np.ndarray) -> np.ndarray:
        out = np.asarray(wav, dtype=np.float32)
        if out.ndim > 1:
            out = out.mean(axis=-1)
        if out.shape[0] < self.min_input_samples:
            out = np.pad(out, (0, self.min_input_samples - out.shape[0]), mode="constant")
        return out

    @torch.inference_mode()
    def transcribe_batch(self, wavs: List[np.ndarray], sample_rate: int) -> List[DecodeResult]:
        t0 = time.perf_counter()
        wavs = [self._prepare_wav(w) for w in wavs]
        inputs = self.processor(
            wavs,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs.input_values.to(self.cfg.device, dtype=self.torch_dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.cfg.device)

        logits = self.model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        texts = self.processor.batch_decode(pred_ids)
        t1 = time.perf_counter()

        audio_sec = sum(len(w) for w in wavs) / float(sample_rate)
        compute_sec = max(1e-9, (t1 - t0))
        rtf = compute_sec / max(1e-9, audio_sec)

        return [DecodeResult(text=t.strip(), rtf=rtf) for t in texts]
