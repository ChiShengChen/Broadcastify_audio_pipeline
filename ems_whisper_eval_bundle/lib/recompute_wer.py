#!/usr/bin/env python3
"""Recompute WER/CER on existing eval_human_annotation.csv files under different
normalization modes so v5/v6 can be compared fairly.

Modes:
  raw       : no normalization
  ems       : ems_eval.preprocessing.normalize_ems_text (lowercase + abbrev + punct)
  whisper   : transformers EnglishTextNormalizer (handles digits<->words, brackets, contractions)
  combined  : ems then whisper

Usage:
  python recompute_wer_normalized.py csv1.csv [csv2.csv ...]
"""
import argparse
import csv
import sys
from pathlib import Path

from jiwer import wer, cer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from ems_eval.preprocessing import normalize_ems_text
except Exception:
    normalize_ems_text = None

WHISPER_NORM = EnglishTextNormalizer({})


import re

# Strip standalone unintelligible markers: [x], (x), bare " x " tokens, [unintelligible]
_X_TOKEN_RE = re.compile(r"\b[xX]\b")
_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def strip_unintelligible(text: str) -> str:
    text = _BRACKET_RE.sub(" ", text)
    text = _X_TOKEN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm(text: str, mode: str) -> str:
    text = text or ""
    if mode == "raw":
        return text
    if mode == "ems":
        return normalize_ems_text(text) if normalize_ems_text else text.lower()
    if mode == "whisper":
        return WHISPER_NORM(strip_unintelligible(text))
    if mode == "combined":
        t = strip_unintelligible(text)
        t = normalize_ems_text(t) if normalize_ems_text else t
        return strip_unintelligible(WHISPER_NORM(t))
    raise ValueError(mode)


def score(refs, preds):
    refs = [r if r.strip() else "x" for r in refs]
    preds = [p if p.strip() else "x" for p in preds]
    return wer(refs, preds), cer(refs, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="eval_human_annotation.csv paths")
    args = ap.parse_args()

    modes = ["raw", "ems", "whisper", "combined"]
    print(f"{'tag':<60} {'n':>4} | " + " | ".join(f"{m:>13}" for m in modes))
    print(f"{'':<60} {'':>4} | " + " | ".join(f"{'WER %':>6} {'CER %':>6}" for _ in modes))

    for fn in args.csvs:
        rows = list(csv.DictReader(open(fn)))
        refs_raw = [(r.get("reference") or "").strip() for r in rows]
        preds_raw = [(r.get("prediction") or "").strip() for r in rows]
        cells = []
        for m in modes:
            refs = [norm(r, m) for r in refs_raw]
            preds = [norm(p, m) for p in preds_raw]
            w, c = score(refs, preds)
            cells.append(f"{w*100:5.1f}% {c*100:5.1f}%")
        tag = Path(fn).parent.name
        print(f"{tag:<60} {len(rows):>4} | " + " | ".join(f"{c:>13}" for c in cells))

        # Show first 2 examples after combined normalization for inspection
        if rows:
            print(f"  --- sample (combined-normalized) ---")
            for r, p in list(zip(refs_raw, preds_raw))[:2]:
                print(f"  REF: {norm(r,'combined')[:120]}")
                print(f"  HYP: {norm(p,'combined')[:120]}")
            print()


if __name__ == "__main__":
    main()
