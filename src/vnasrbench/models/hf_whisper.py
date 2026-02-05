from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from vnasrbench.models.base import ASRModel, DecodeResult


@dataclass
class HFWhisperConfig:
    hf_id: str
    language: str = "vi"
    task: str = "transcribe"
    device: str = "cuda"
    dtype: str = "float16"
    batch_size: int = 4


class HFWhisperModel(ASRModel):
    def __init__(self, cfg: HFWhisperConfig) -> None:
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.hf_id)
        torch_dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(cfg.hf_id, torch_dtype=torch_dtype)
        self.model.to(cfg.device)
        self.model.eval()

    @torch.inference_mode()
    def transcribe_batch(self, wavs: List[np.ndarray], sample_rate: int) -> List[DecodeResult]:
        t0 = time.perf_counter()
        inputs = self.processor(
            wavs,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(self.cfg.device)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.cfg.language, task=self.cfg.task
        )
        gen = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
        )
        texts = self.processor.batch_decode(gen, skip_special_tokens=True)
        t1 = time.perf_counter()

        # RTF: compute_time / total_audio_time
        audio_sec = sum(len(w) for w in wavs) / float(sample_rate)
        compute_sec = max(1e-9, (t1 - t0))
        rtf = compute_sec / max(1e-9, audio_sec)

        return [DecodeResult(text=t.strip(), rtf=rtf) for t in texts]
