from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Set, Tuple
import hashlib

import numpy as np


@dataclass(frozen=True)
class HFDatasetConfig:
    hf_id: str
    split: str
    text_key: str
    id_key: Optional[str] = None
    streaming: bool = True
    limit: Optional[int] = None
    id_list_path: Optional[str] = None
    revision: Optional[str] = None

    def load_id_set(self) -> Optional[Set[str]]:
        if not self.id_list_path:
            return None
        ids: Set[str] = set()
        for line in Path(self.id_list_path).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.add(s)
        return ids


def compute_id_list_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def iter_hf_dataset(cfg: HFDatasetConfig) -> Iterator[Tuple[str, str, np.ndarray, int]]:
    """
    Stream examples from a Hugging Face dataset and yield:
    (utt_id, text, audio_array, sample_rate)
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: datasets[audio]. "
            "Install with: pip install -U \"datasets[audio]\""
        ) from e

    ds = load_dataset(
        cfg.hf_id,
        split=cfg.split,
        streaming=cfg.streaming,
        revision=cfg.revision,
    )
    id_set = cfg.load_id_set()

    n = 0
    for i, ex in enumerate(ds):
        utt_id = None
        if cfg.id_key and cfg.id_key in ex:
            utt_id = ex[cfg.id_key]
        else:
            utt_id = ex.get("utt_id") or ex.get("id") or f"{cfg.split}_{i:08d}"
        utt_id = str(utt_id)

        if id_set is not None and utt_id not in id_set:
            continue

        audio = ex["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        text = str(ex[cfg.text_key])

        yield utt_id, text, arr, sr
        n += 1
        if cfg.limit is not None and n >= cfg.limit:
            break
