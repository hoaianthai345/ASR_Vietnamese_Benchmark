from __future__ import annotations

import re

# Strict normalization for WER: keep Vietnamese diacritics + numerals, drop punctuation.
_PUNCT_STRICT = re.compile(r"[,\.\!\?\:\;\-\—\(\)\[\]\"\'\“\”\‘\’]")
_SPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")

# A lighter normalization used for error taxonomy and stitching: keep punctuation but normalize spacing.
_PUNCT_SPACING = re.compile(r"([,\.\!\?\:\;])")


def normalize_vi(text: str) -> str:
    # Backward-compatible alias for strict WER normalization.
    return normalize_wer_strict_vi(text)


def normalize_wer_strict_vi(text: str) -> str:
    t = text.lower().strip()
    t = _PUNCT_STRICT.sub(" ", t)
    t = _SPACE.sub(" ", t).strip()
    return t


def normalize_cer_strict_vi(text: str) -> str:
    # Use strict normalization, then remove spaces to reduce tokenization sensitivity.
    t = normalize_wer_strict_vi(text)
    return t.replace(" ", "")


def normalize_taxonomy_vi(text: str) -> str:
    # Keep punctuation so we can attribute punctuation-related errors.
    # We still lowercase + canonicalize spacing to reduce noise.
    t = text.lower().strip()
    t = _PUNCT_SPACING.sub(r" \1 ", t)
    t = _SPACE.sub(" ", t).strip()
    return t


def mask_numbers(text: str, token: str = "<num>") -> str:
    # Conservative number normalization for ablation: collapse any digit span to a single token.
    return _DIGITS.sub(token, text)
