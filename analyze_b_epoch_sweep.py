"""Compare base vs B_5ep vs B_1ep on aligned VB eval."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/media/meow/One Touch/ems_call")
sys.path.insert(0, str(ROOT))
from bootstrap_wer_ci import per_clip_stats, bootstrap_ci

vb = pd.read_csv(ROOT / "vb_ems_anotation" / "human_anotation_vb.csv")
vb = vb[vb["transcript"].notna() & (vb["transcript"].str.strip() != "")]

csvs = {
    "base": ROOT / "vb_aligned_eval" / "base_whisper_large_v3.csv",
    "B_5ep": ROOT / "speaker_aug_sweep_clean" / "B_clean_20spk_orig_on_aligned.csv",
    "B_1ep": ROOT / "speaker_aug_sweep_clean" / "B_clean_20spk_orig_1ep_on_aligned.csv",
}
stats = {k: per_clip_stats(p, vb) for k, p in csvs.items()}
res = bootstrap_ci(stats, n_iter=10000, seed=42, alpha=0.05, base_key="base")

print(f"{'cond':<10s}  {'WER':>6s}  {'95% CI':>16s}  {'Δ vs base':>10s}  {'95% Δ-CI':>17s}  {'>100%':>6s}  {'worst':>7s}")
print("-" * 92)
for k in csvs:
    r = res[k]
    ci = f"[{r['ci_lo']*100:5.2f}, {r['ci_hi']*100:5.2f}]"
    rows = [(c, (e/ref if ref else 0.0)) for c, e, ref in stats[k]]
    n100 = sum(1 for _, w in rows if w > 1.0)
    worst = max(w for _, w in rows) * 100
    if k == "base":
        print(f"{k:<10s}  {r['wer']*100:5.2f}%  {ci:>16s}  {'':>10s}  {'':>17s}  {n100:>3d}/{len(rows):<2d}  {worst:>6.1f}%")
    else:
        d_ci = f"[{r['diff_lo']*100:+5.2f}, {r['diff_hi']*100:+5.2f}]"
        print(f"{k:<10s}  {r['wer']*100:5.2f}%  {ci:>16s}  {r['diff_vs_base']*100:+8.2f}pp  {d_ci:>17s}  {n100:>3d}/{len(rows):<2d}  {worst:>6.1f}%")

# Compare worst tail clips between 5ep and 1ep
print("\n=== Per-clip Δ (B_1ep − B_5ep), top-5 most improved by lowering epoch ===")
b5 = {c: e/r if r else 0 for c, e, r in stats["B_5ep"]}
b1 = {c: e/r if r else 0 for c, e, r in stats["B_1ep"]}
deltas = sorted(((c, b1[c] - b5[c]) for c in b5 if c in b1), key=lambda x: x[1])
for c, d in deltas[:5]:
    print(f"  {c[:48]:<48s}  5ep={b5[c]*100:5.1f}%  1ep={b1[c]*100:5.1f}%  Δ={d*100:+.1f}pp")
print("\n=== Top-5 worsened by lowering epoch ===")
for c, d in deltas[-5:][::-1]:
    print(f"  {c[:48]:<48s}  5ep={b5[c]*100:5.1f}%  1ep={b1[c]*100:5.1f}%  Δ={d*100:+.1f}pp")
