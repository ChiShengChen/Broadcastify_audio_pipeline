# EMS Whisper Evaluation Bundle

Run our 3 fine-tuned Whisper models (and a vanilla baseline) on **your** EMS audio and transcripts. You only need to point the script at two paths.

## What's in this bundle

```
ems_whisper_eval_bundle/
├── run_evaluation.py          ← Entry #1: WER evaluation on your wavs+transcripts
├── run_gap_analysis.py        ← Entry #2: domain-gap analysis (your real vs our synthetic)
├── requirements.txt           ← Python deps (pip install)
├── README.md                  ← You are here
├── models/                    ← LoRA adapters (~64 MB each)
│   ├── v5_single/             ← Fine-tuned on 102 synthetic wavs (VoxCPM raw + radio)
│   ├── v6_single/             ← Fine-tuned on 822 synthetic wavs (8× v5)
│   └── v6_aug/                ← Fine-tuned on 1644 radio-domain-augmented wavs (4× aug)
├── reference_data/            ← 50 sampled v6 wavs per variant (16k mono, ~135 MB)
│   ├── v6_raw/                ← Pure VoxCPM TTS, no radio simulation
│   ├── v6_radio/              ← VoxCPM TTS + VoxCPM's own radio simulation
│   └── v6_aug/                ← Our 4× radio-domain audio augmentation
├── lib/                       ← Internal scripts (you don't need to touch)
│   ├── prepare_dataset.py
│   ├── evaluate.py
│   ├── recompute_wer.py
│   ├── analyze_gap.py
│   └── ems_eval/              ← EMS abbreviation expansion + medical vocab
└── sample_data/
    └── sample_transcripts.csv ← Format reference
```

The base model `openai/whisper-large-v3` (~3 GB) is downloaded automatically from HuggingFace Hub on first run; not bundled.

## Requirements

- **Python 3.10+** (tested on 3.11)
- **GPU recommended** (whisper-large-v3 needs ~6 GB FP16 / 12 GB FP32). CPU works but each clip takes ~30 s instead of ~3 s.
- ~5 GB free disk for HuggingFace model cache

## Setup

```bash
# 1. (Optional but recommended) create a fresh venv / conda env
python -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt
```

## Usage

You need **two inputs**:

1. **A folder of `.wav` files** (16 kHz mono recommended; the script will resample if needed)
2. **A CSV with columns `Filename, transcript`** — one row per wav. Other columns are ignored. See [`sample_data/sample_transcripts.csv`](sample_data/sample_transcripts.csv) for the exact format.

Then run:

```bash
python run_evaluation.py \
  --wav_dir       /path/to/your/wav_folder \
  --transcript_csv /path/to/your/transcripts.csv
```

Optional flags:
- `--output_dir /custom/results/path` (default: `./results/`)
- `--models baseline v6_single` (subset of models; default: all four)

## What you'll get

Inside `results/` (or your `--output_dir`):

```
results/
├── _test_dataset/               # HuggingFace Dataset cache (auto-cleanable)
├── baseline_predictions.csv     # per-clip: original_file, reference, prediction
├── v5_single_predictions.csv
├── v6_single_predictions.csv
├── v6_aug_predictions.csv
└── summary.csv                  # WER/CER per model under 4 normalization modes
```

`summary.csv` and the console output look like:

```
model           n | raw WER raw CER | ems WER ems CER | whisper WER whisper CER | combined WER combined CER
baseline       50 |  88.3%   75.5% |   82.9%  73.9% |    81.7%   73.3% |    81.9%   73.4%
v5_single      50 |  88.3%   75.6% |   83.0%  74.0% |    81.7%   73.4% |    81.9%   73.5%
v6_single      50 |  89.0%   75.5% |   83.3%  73.6% |    82.3%   73.0% |    82.5%   73.1%
v6_aug         50 |  91.1%   77.0% |   86.8%  75.2% |    85.3%   75.0% |    85.4%   75.0%
```

### Which WER column should I look at?

**`combined` is the fairest** — it normalizes both reference and prediction with:
1. EMS abbreviation expansion (e.g. `pt` → `patient`, `bp` → `blood pressure`)
2. Whisper's `EnglishTextNormalizer`, which canonicalizes digit ↔ word forms (`"eighty-two"` ↔ `"82"`), removes `[x]` markers, expands contractions, and strips punctuation/case.

Why this matters: our v6 training data uses spelled-out numbers (`"eighty-two-year-old"`) while real EMS annotations typically use digits (`"82 year old"`). Without normalization, this format mismatch alone inflates raw WER by ~10 pp.

## Optional: domain-gap analysis

If you'd also like to see *how acoustically different* your Harvard EMS data is from our synthetic v6 data (this helps explain why a fine-tuned model may or may not generalize), run:

```bash
python run_gap_analysis.py --real_wav_dir /path/to/your/wav_folder
```

You only need to provide your real radio wav folder — the 50 reference v6 wavs (raw / radio / aug) are already bundled inside `reference_data/`.

This computes:

1. **DSP statistics** per set: spectral centroid, F0 (pitch) std, pause %, bandwidth, etc. Saved to `gap_results/gap_stats.csv`.
2. **Whisper-encoder embedding distances**: how far apart your data is from each v6 variant in the ASR model's internal representation (centroid distance, mean pairwise, MMD-RBF). Saved to `gap_results/gap_distances.csv`.
3. **t-SNE plot** of all 4 sets (real / v6_raw / v6_radio / v6_aug) projected to 2D. Saved to `gap_results/gap_tsne.png`.

### What the numbers mean

- **Pause %** — fraction of silent frames. Real EMS radio is often >50% silence (squelch / mic-key gaps). Our synthetic is 1–10%. Big gap here = real-world radio characteristic that synthetic lacks.
- **Spectral centroid (Hz)** — "center of mass" of the spectrum. Real radio is bandlimited (300–3400 Hz) so centroid ~500 Hz. Synthetic without bandpass is ~1800 Hz.
- **F0 std (Hz)** — pitch variability (intonation). Real dispatcher speech is flatter (~50 Hz std) than read-style TTS (~85 Hz std).
- **Wasserstein distance** — distribution distance per feature. Larger = more different.
- **Whisper-encoder MMD-RBF** — most ASR-relevant. If MMD between your real and any v6 variant is < ~0.3, fine-tune may help; > 0.5 means the model "sees" them as different distributions and fine-tune may not transfer.

Our own real (50 clips) vs v6 reference numbers (for comparison):

| pair | centroid_dist | mean_pairwise | MMD-RBF |
|---|---:|---:|---:|
| our_real vs v6_raw | 4.65 | 5.58 | 0.67 |
| our_real vs v6_radio | 4.63 | 5.58 | 0.66 |
| our_real vs v6_aug | 4.13 | 5.29 | 0.54 |

If your numbers are *closer* than ours, it suggests our v6 data is more similar to your Harvard radio than to our own — fine-tune may help on your data even if it didn't help on ours.

---

## Background (our results on n=50 of our own data)

| Model | combined WER | combined CER |
|---|---:|---:|
| baseline (no fine-tune) | **81.9%** ⭐ | 73.4% |
| v5_single | 81.9% | 73.5% |
| v6_single | 82.5% | 73.1% |
| v6_aug | 85.4% | 75.0% |

On our test set of 50 real EMS radio clips, **none of the synthetic-data fine-tunes outperform the baseline**, and the radio-domain audio augmentation actively hurts (−3.5 pp). We'd love to see whether the same pattern holds on your Harvard EMS data, or whether your domain is closer to one of the synthetic configurations.

## Troubleshooting

- **`CUDA out of memory`**: run one model at a time with `--models baseline`, then `--models v6_single`, etc.
- **`No module named 'peft'`**: re-run `pip install -r requirements.txt`. PEFT is required to load LoRA adapters.
- **`boto3.__spec__ is None`**: a `boto3` install is partially broken. Run `pip install --force-reinstall boto3`.
- **WAV files not found**: the script matches CSV `Filename` exactly against the basename of files in `--wav_dir` (no recursion). Check both sides match (e.g. case sensitivity, .wav vs .WAV).

## Reference

Full methodology and per-experiment numbers in our internal report:
`results_v5_v6_baseline_20260505/EMS_V5_V6_BASELINE_RESULTS.md`

Contact: [you@example.com] for questions or to share back results.
