from __future__ import annotations

import re

_punct = re.compile(r"[\,\.\!\?\:\;\-\—\(\)\[\]"'\“\”\‘\’]")

_space = re.compile(r"\s+")


def normalize_vi(text: str) -> str:
    t = text.lower().strip()
    t = _punct.sub(" ", t)
    t = _space.sub(" ", t).strip()
    return t
