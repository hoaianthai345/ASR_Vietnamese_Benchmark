from __future__ import annotations

from pathlib import Path
from typing import Tuple
import soundfile as sf

from vnasrbench.data.manifest import read_jsonl


def validate_manifest(manifest_path: str) -> Tuple[int, int]:
    missing = 0
    total = 0
    for u in read_jsonl(manifest_path):
        total += 1
        if not Path(u.audio_path).exists():
            missing += 1
            continue
        # quick decode check
        sf.info(u.audio_path)
        if not u.text.strip():
            raise ValueError(f"Empty text for utt_id={u.utt_id}")
    return total, missing
