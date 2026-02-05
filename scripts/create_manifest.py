from __future__ import annotations

import argparse
from pathlib import Path
import soundfile as sf

from vnasrbench.data.manifest import Utterance, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--transcript", required=True, help="TSV: utt_id\ttext OR txt: utt_id|text")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    tpath = Path(args.transcript)

    mapping = {}
    lines = tpath.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "	" in line:
            utt_id, text = line.split("	", 1)
        elif "|" in line:
            utt_id, text = line.split("|", 1)
        else:
            raise ValueError("Transcript line must contain tab or |")
        mapping[utt_id.strip()] = text.strip()

    items = []
    for utt_id, text in mapping.items():
        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            continue
        info = sf.info(str(wav_path))
        dur = float(info.frames) / float(info.samplerate)
        items.append(Utterance(utt_id=utt_id, audio_path=str(wav_path), text=text, duration_sec=dur))

    write_jsonl(args.out, items)
    print(f"Wrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
