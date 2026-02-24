from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from tqdm import tqdm


def dump_manifest(
    hf_id: str,
    split: str,
    out_path: str,
    text_key: str,
    dataset_tag: str,
    id_key: Optional[str] = None,
    limit: Optional[int] = None,
    streaming: bool = True,
) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: datasets[audio]. Install with: pip install -U \"datasets[audio]\""
        ) from e

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds = load_dataset(hf_id, split=split, streaming=streaming)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in tqdm(ds):
            audio = ex["audio"]
            wav = audio["array"]
            sr = audio["sampling_rate"]
            dur = len(wav) / float(sr)

            utt_id = None
            if id_key and id_key in ex:
                utt_id = ex[id_key]
            else:
                utt_id = ex.get("utt_id") or ex.get("id") or f"{dataset_tag}_{n:08d}"

            text = ex[text_key]

            f.write(
                json.dumps(
                    {
                        "utt_id": str(utt_id),
                        "audio_path": "",
                        "text": str(text),
                        "duration_sec": float(dur),
                        "hf_dataset": hf_id,
                        "hf_split": split,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
            if limit and n >= limit:
                break
    print("Wrote", n, "rows to", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--text_key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--id_key", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-streaming", action="store_true")
    args = ap.parse_args()

    dump_manifest(
        hf_id=args.hf_id,
        split=args.split,
        out_path=args.out,
        text_key=args.text_key,
        dataset_tag=args.dataset_tag,
        id_key=args.id_key or None,
        limit=args.limit or None,
        streaming=not args.no_streaming,
    )


if __name__ == "__main__":
    main()
