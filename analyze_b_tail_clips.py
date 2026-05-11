"""Show ref + base-hyp + B-hyp for the 3 clips where B has WER > 100%."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/media/meow/One Touch/ems_call")
sys.path.insert(0, str(ROOT))
from bootstrap_wer_ci import per_clip_stats, lev_errors
from ems_eval.preprocessing import normalize_ems_text


def joined_pred_per_clip(csv_path):
    df = pd.read_csv(csv_path)
    out = {}
    for src, grp in df.groupby("source_file", sort=False):
        out[src] = " ".join(str(p) for p in grp["prediction"])
    return out


def main():
    vb = pd.read_csv(ROOT / "vb_ems_anotation" / "human_anotation_vb.csv")
    vb = vb[vb["transcript"].notna() & (vb["transcript"].str.strip() != "")]
    ref_raw = dict(zip(vb["Filename"], vb["transcript"].astype(str)))

    base_csv = ROOT / "vb_aligned_eval" / "base_whisper_large_v3.csv"
    b_csv = ROOT / "speaker_aug_sweep_clean" / "B_clean_20spk_orig_on_aligned.csv"

    base_pred = joined_pred_per_clip(base_csv)
    b_pred = joined_pred_per_clip(b_csv)
    b_stats = per_clip_stats(b_csv, vb)

    # Find clips where B's WER > 1.0
    bad = [(c, e, r, e / r) for c, e, r in b_stats if r and (e / r) > 1.0]
    bad.sort(key=lambda x: -x[3])
    print(f"B clips with WER > 100%: {len(bad)}\n")

    for clip, e, r, w in bad:
        ref_txt = ref_raw.get(clip, "")
        base_txt = base_pred.get(clip, "")
        b_txt = b_pred.get(clip, "")
        ref_norm = normalize_ems_text(ref_txt).split()
        base_norm = normalize_ems_text(base_txt).split()
        b_norm = normalize_ems_text(b_txt).split()
        e_base = lev_errors(ref_norm, base_norm)
        wer_base = e_base / len(ref_norm) if ref_norm else 0
        print("=" * 100)
        print(f"CLIP: {clip}")
        print(f"  ref tokens: {len(ref_norm)}  |  base hyp tokens: {len(base_norm)}  |  B hyp tokens: {len(b_norm)}")
        print(f"  base WER: {wer_base*100:.1f}%  |  B WER: {w*100:.1f}%  |  Δ: {(w-wer_base)*100:+.1f}pp")
        print(f"\n[REF raw]\n{ref_txt}")
        print(f"\n[REF normalized]\n{' '.join(ref_norm)}")
        print(f"\n[BASE hyp raw]\n{base_txt}")
        print(f"\n[B hyp raw]\n{b_txt}")
        print()


if __name__ == "__main__":
    main()
