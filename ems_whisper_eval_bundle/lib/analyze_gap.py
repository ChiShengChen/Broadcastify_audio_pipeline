"""Quantify acoustic / encoder-embedding gap between collaborator's real radio
recordings and our bundled v6 synthetic reference samples.

Outputs:
  <out>/gap_stats.csv       per-set DSP statistics (mean/std)
  <out>/gap_distances.csv   Whisper-encoder centroid / pairwise / MMD distances
  <out>/gap_tsne.png        2D t-SNE plot of encoder embeddings
"""
import argparse
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# boto3 stub guard (same as evaluate.py)
if "boto3" not in sys.modules:
    try:
        import boto3 as _real_boto3  # noqa: F401
    except Exception:
        _b = types.ModuleType("boto3"); _bs = types.ModuleType("boto3.session")
        _bs.Session = type("DummySession", (), {})
        _b.session = _bs
        sys.modules["boto3"] = _b
        sys.modules["boto3.session"] = _bs

import torch
from transformers import WhisperProcessor, WhisperModel
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance


SR = 16000
random.seed(42)


def collect_wavs(d: Path) -> list:
    return sorted(p for p in d.iterdir() if p.suffix.lower() == ".wav") if d.is_dir() else []


def load_audio(path: Path, sr: int = SR, max_seconds: float = 25.0) -> np.ndarray:
    a, file_sr = sf.read(path)
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32)
    if file_sr != sr:
        a = librosa.resample(a, orig_sr=file_sr, target_sr=sr)
    if len(a) > int(max_seconds * sr):
        a = a[: int(max_seconds * sr)]
    return a


def dsp_stats(a: np.ndarray, sr: int = SR) -> dict:
    if len(a) < sr:
        return {}
    rms = float(np.sqrt((a ** 2).mean() + 1e-12))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=a, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=a, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=a, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=a)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=a)))

    e = librosa.feature.rms(y=a, frame_length=1024, hop_length=512)[0]
    pause_pct = float((e < max(e.mean() * 0.2, 1e-4)).mean())

    try:
        f0 = librosa.yin(a, fmin=70, fmax=400, sr=sr)
        f0 = f0[~np.isnan(f0)]
        f0 = f0[(f0 > 70) & (f0 < 400)]
        f0_mean = float(np.mean(f0)) if len(f0) > 0 else 0.0
        f0_std = float(np.std(f0)) if len(f0) > 0 else 0.0
    except Exception:
        f0_mean = f0_std = 0.0

    S = np.abs(librosa.stft(a, n_fft=1024, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    mask = (freqs >= 300) & (freqs <= 3400)
    e_radio = float((S[mask] ** 2).sum())
    e_total = float((S ** 2).sum() + 1e-12)
    return {
        "rms": rms, "centroid_hz": centroid, "bandwidth_hz": bandwidth,
        "rolloff_hz": rolloff, "flatness": flatness, "zcr": zcr,
        "pause_pct": pause_pct, "f0_mean_hz": f0_mean, "f0_std_hz": f0_std,
        "radio_band_ratio": e_radio / e_total, "duration_s": len(a) / sr,
    }


def whisper_embed(audios: list, processor: WhisperProcessor, model: WhisperModel,
                  device: str) -> np.ndarray:
    embs = []
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for a in audios:
            inp = processor.feature_extractor(a, sampling_rate=SR, return_tensors="pt").input_features.to(device)
            if dtype == torch.float16:
                inp = inp.half()
            out = model.encoder(inp).last_hidden_state.mean(dim=1)
            embs.append(out.float().cpu().numpy()[0])
    return np.stack(embs)


def mmd2(X: np.ndarray, Y: np.ndarray) -> float:
    sigma = np.median(np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)) + 1e-9
    def k(a, b):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2 * sigma ** 2))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def run(real_dir: Path, ref_root: Path, out_dir: Path, n: int = 50,
        whisper_model: str = "openai/whisper-large-v3"):
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = {
        "real":     real_dir,
        "v6_raw":   ref_root / "v6_raw",
        "v6_radio": ref_root / "v6_radio",
        "v6_aug":   ref_root / "v6_aug",
    }

    selections = {}
    for s, d in sets.items():
        files = collect_wavs(d)
        if not files:
            print(f"  ⚠️  {s}: no wavs in {d}")
            continue
        selections[s] = random.sample(files, min(n, len(files)))
        print(f"  {s}: {len(files)} candidates, sampled {len(selections[s])}")
    if "real" not in selections or len(selections.get("real", [])) < 5:
        raise SystemExit("Need at least 5 wavs in --real_wav_dir")

    # 1) DSP stats
    rows, audios_by_set = [], {s: [] for s in selections}
    for s, files in selections.items():
        for f in files:
            try:
                a = load_audio(f)
            except Exception as e:
                print(f"  skip {f}: {e}"); continue
            stats = dsp_stats(a)
            stats.update({"set": s, "file": f.name})
            rows.append(stats)
            audios_by_set[s].append(a)
    df = pd.DataFrame(rows)
    summary = df.drop(columns=["file"]).groupby("set").agg(["mean", "std"]).round(3)
    print("\n=== DSP statistics (mean ± std per set) ===")
    print(summary.to_string())
    summary.to_csv(out_dir / "gap_stats.csv")

    print("\n=== Wasserstein distance (real vs each synthetic) ===")
    feats = ["rms", "centroid_hz", "bandwidth_hz", "rolloff_hz",
             "f0_mean_hz", "f0_std_hz", "pause_pct", "radio_band_ratio"]
    real = df[df["set"] == "real"]
    others = [s for s in selections if s != "real"]
    print(f"{'feature':<20}" + "".join(f"{s:>14}" for s in others))
    for feat in feats:
        line = f"{feat:<20}"
        for s in others:
            other = df[df["set"] == s][feat].values
            wd = wasserstein_distance(real[feat].values, other)
            line += f"{wd:>14.4f}"
        print(line)

    # 2) Whisper encoder embedding
    print(f"\n=== Whisper encoder embedding ({whisper_model}) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {whisper_model} on {device} ...")
    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperModel.from_pretrained(whisper_model).to(device).eval()
    if device == "cuda":
        model = model.half()

    embs_by_set = {}
    for s in selections:
        print(f"  embedding {s} ({len(audios_by_set[s])} clips)...")
        embs_by_set[s] = whisper_embed(audios_by_set[s], processor, model, device)

    print(f"\n{'pair':<22}{'centroid_dist':>16}{'mean_pairwise':>16}{'mmd_rbf':>12}")
    real_emb = embs_by_set["real"]
    dist_rows = []
    for s in others:
        emb = embs_by_set[s]
        c_dist = float(np.linalg.norm(real_emb.mean(0) - emb.mean(0)))
        ra = real_emb if len(real_emb) <= 100 else real_emb[np.random.choice(len(real_emb), 100, replace=False)]
        ea = emb if len(emb) <= 100 else emb[np.random.choice(len(emb), 100, replace=False)]
        pair = float(np.linalg.norm(ra[:, None, :] - ea[None, :, :], axis=-1).mean())
        m = mmd2(ra, ea)
        print(f"real vs {s:<14}{c_dist:>16.3f}{pair:>16.3f}{m:>12.4f}")
        dist_rows.append({"pair": f"real_vs_{s}", "centroid_dist": c_dist,
                          "mean_pairwise": pair, "mmd_rbf": m})
    pd.DataFrame(dist_rows).to_csv(out_dir / "gap_distances.csv", index=False)

    # 3) t-SNE
    print("\n=== t-SNE (2D) ===")
    all_embs = np.vstack([embs_by_set[s] for s in selections])
    labels = sum([[s] * len(embs_by_set[s]) for s in selections], [])
    perp = min(30, max(5, len(all_embs) // 4))
    proj = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca").fit_transform(all_embs)

    plt.figure(figsize=(8, 6))
    colors = {"real": "tab:red", "v6_raw": "tab:blue", "v6_radio": "tab:green", "v6_aug": "tab:orange"}
    markers = {"real": "o", "v6_raw": "^", "v6_radio": "s", "v6_aug": "D"}
    for s in selections:
        idx = [i for i, l in enumerate(labels) if l == s]
        plt.scatter(proj[idx, 0], proj[idx, 1], c=colors[s], marker=markers[s],
                    label=s, alpha=0.7, s=40, edgecolors="black", linewidths=0.5)
    plt.title("t-SNE of Whisper encoder embeddings\n(closer = ASR sees them as similar)")
    plt.legend(); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.tight_layout()
    plt.savefig(out_dir / "gap_tsne.png", dpi=150)
    print(f"  saved → {out_dir / 'gap_tsne.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_wav_dir", required=True)
    ap.add_argument("--reference_root", required=True,
                    help="Path to reference_data/ containing v6_raw, v6_radio, v6_aug subfolders")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--whisper_model", default="openai/whisper-large-v3")
    args = ap.parse_args()
    run(Path(args.real_wav_dir), Path(args.reference_root),
        Path(args.output_dir), args.n, args.whisper_model)
