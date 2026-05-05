#!/usr/bin/env python3
"""End-to-end evaluation: collaborator provides --wav_dir + --transcript_csv,
runs all bundled models (baseline + v5_single + v6_single + v6_aug), and
emits a single comparison summary.

Example:
  python run_evaluation.py \\
      --wav_dir /path/to/your/wav_folder \\
      --transcript_csv /path/to/your/transcripts.csv

Transcript CSV must have columns: Filename, transcript
(other columns like Call Type / Tags / Notes are optional and ignored)
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_ROOT / "lib"))

from prepare_dataset import build_dataset
from evaluate import evaluate
from recompute_wer import norm, score
from jiwer import wer as wer_fn, cer as cer_fn  # noqa: F401


MODELS = [
    ("baseline",  "openai/whisper-large-v3"),
    ("v5_single", str(BUNDLE_ROOT / "models" / "v5_single")),
    ("v6_single", str(BUNDLE_ROOT / "models" / "v6_single")),
    ("v6_aug",    str(BUNDLE_ROOT / "models" / "v6_aug")),
]
NORM_MODES = ["raw", "ems", "whisper", "combined"]


def main():
    ap = argparse.ArgumentParser(description="Run all bundled Whisper models on collaborator's data")
    ap.add_argument("--wav_dir", required=True, help="Directory containing the .wav files")
    ap.add_argument("--transcript_csv", required=True,
                    help="CSV with columns: Filename, transcript")
    ap.add_argument("--output_dir", default=str(BUNDLE_ROOT / "results"),
                    help="Where to write predictions + summary (default: ./results/)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of models to run, e.g. --models baseline v6_single (default: all)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build HF dataset from collaborator's data
    dataset_dir = out_dir / "_test_dataset"
    print(f"\n=== [1/3] Building test dataset from {args.transcript_csv} + {args.wav_dir} ===")
    build_dataset(args.wav_dir, args.transcript_csv, str(dataset_dir))

    # 2. Run each model
    selected = args.models or [name for name, _ in MODELS]
    csv_paths = []
    for name, model_path in MODELS:
        if name not in selected:
            continue
        out_csv = out_dir / f"{name}_predictions.csv"
        print(f"\n=== [2/3] Evaluating {name} ===")
        evaluate(model_path, str(dataset_dir), str(out_csv))
        csv_paths.append((name, out_csv))

    # 3. Recompute WER under all normalization modes + write summary
    print(f"\n=== [3/3] Computing WER summary ===")
    summary_rows = []
    for name, p in csv_paths:
        rows = list(csv.DictReader(open(p)))
        refs_raw = [(r.get("reference") or "").strip() for r in rows]
        preds_raw = [(r.get("prediction") or "").strip() for r in rows]
        record = {"model": name, "n": len(rows)}
        for m in NORM_MODES:
            refs = [norm(r, m) for r in refs_raw]
            preds = [norm(p, m) for p in preds_raw]
            w, c = score(refs, preds)
            record[f"WER_{m}_pct"] = round(w * 100, 2)
            record[f"CER_{m}_pct"] = round(c * 100, 2)
        summary_rows.append(record)

    summary_csv = out_dir / "summary.csv"
    fields = ["model", "n"] + [f"{m}_{k}_pct" for m in ("WER", "CER") for k in NORM_MODES]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    # Console table
    print(f"\nSaved summary → {summary_csv}\n")
    print(f"{'model':<14} {'n':>4} | " + " | ".join(f"{m:>13}" for m in NORM_MODES))
    print(f"{'':<14} {'':>4} | " + " | ".join(f"{'WER %':>6} {'CER %':>6}" for _ in NORM_MODES))
    for r in summary_rows:
        cells = [f"{r[f'WER_{m}_pct']:5.1f}% {r[f'CER_{m}_pct']:5.1f}%" for m in NORM_MODES]
        print(f"{r['model']:<14} {r['n']:>4} | " + " | ".join(f"{c:>13}" for c in cells))

    print(f"\nPer-clip predictions:")
    for name, p in csv_paths:
        print(f"  {name}: {p}")
    print()


if __name__ == "__main__":
    main()
