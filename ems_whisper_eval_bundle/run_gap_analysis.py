#!/usr/bin/env python3
"""Compute the acoustic and Whisper-encoder gap between collaborator's real
EMS radio recordings and our bundled v6 synthetic samples (raw / radio / aug).

The 50 reference v6 wavs per variant are bundled inside reference_data/, so the
collaborator only needs to provide their own real radio wav folder.

Example:
  python run_gap_analysis.py --real_wav_dir /path/to/your/wav_folder
"""
import argparse
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_ROOT / "lib"))

from analyze_gap import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_wav_dir", required=True,
                    help="Folder of your real EMS radio .wav files")
    ap.add_argument("--output_dir", default=str(BUNDLE_ROOT / "gap_results"))
    ap.add_argument("--n", type=int, default=50,
                    help="Number of clips to sample per set (default 50)")
    ap.add_argument("--whisper_model", default="openai/whisper-large-v3")
    args = ap.parse_args()

    ref_root = BUNDLE_ROOT / "reference_data"
    if not ref_root.is_dir():
        sys.exit(f"reference_data/ not found under {BUNDLE_ROOT}")

    run(Path(args.real_wav_dir), ref_root, Path(args.output_dir),
        args.n, args.whisper_model)
    print(f"\nDone. Outputs in: {args.output_dir}")
    print("  gap_stats.csv      DSP statistics per set")
    print("  gap_distances.csv  Whisper encoder distances (real vs each v6 variant)")
    print("  gap_tsne.png       t-SNE visualization")


if __name__ == "__main__":
    main()
