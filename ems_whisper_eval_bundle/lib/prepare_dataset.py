"""Build a HuggingFace Dataset from a transcript CSV + WAV folder.

Required CSV columns:
  - Filename: relative wav file name (matched against wavs in --wav_dir)
  - transcript: ground-truth text

Optional columns (preserved if present): Call Type, Tags, Notes
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, Audio


def build_dataset(wav_dir: str, csv_path: str, output_dir: str,
                  target_sr: int = 16000) -> str:
    df = pd.read_csv(csv_path)
    if "Filename" not in df.columns or "transcript" not in df.columns:
        raise ValueError("CSV must have 'Filename' and 'transcript' columns")

    df = df[df["transcript"].notna() & (df["transcript"].astype(str).str.strip() != "")]
    print(f"  {len(df)} rows with non-empty transcript")

    # Index wavs in the directory (top-level only — no recursion)
    wav_index = {}
    for entry in os.listdir(wav_dir):
        if entry.lower().endswith(".wav"):
            wav_index[entry] = os.path.join(wav_dir, entry)

    rows = {"audio": [], "transcript": [], "original_file": [], "segment_index": []}
    missing = []
    for _, r in df.iterrows():
        fn = r["Filename"]
        path = wav_index.get(fn)
        if not path:
            missing.append(fn)
            continue
        rows["audio"].append(path)
        rows["transcript"].append(str(r["transcript"]).strip())
        rows["original_file"].append(fn)
        rows["segment_index"].append(0)

    print(f"  matched {len(rows['audio'])}/{len(df)} wavs in {wav_dir}")
    if missing:
        print(f"  missing {len(missing)} (first 5: {missing[:5]})")

    ds = Dataset.from_dict(rows).cast_column("audio", Audio(sampling_rate=target_sr))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(output_dir)
    print(f"  saved → {output_dir} (n={len(ds)})")
    return output_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--transcript_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_sr", type=int, default=16000)
    args = ap.parse_args()
    build_dataset(args.wav_dir, args.transcript_csv, args.output_dir, args.target_sr)
