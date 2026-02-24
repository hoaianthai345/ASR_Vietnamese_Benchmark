from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import json


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    audio_path: str
    text: str
    duration_sec: float


def read_jsonl(path: str | Path) -> Iterator[Utterance]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield Utterance(
                utt_id=str(obj["utt_id"]),
                audio_path=str(obj["audio_path"]),
                text=str(obj["text"]),
                duration_sec=float(obj.get("duration_sec", -1.0)),
            )


def write_jsonl(path: str | Path, items: Iterable[Utterance]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(
                json.dumps(
                    {
                        "utt_id": it.utt_id,
                        "audio_path": it.audio_path,
                        "text": it.text,
                        "duration_sec": it.duration_sec,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
