#!/usr/bin/env bash
# Stage the 50 annotated VB eval clips into /tmp/vb_real (as symlinks to the
# files in random_samples_1/2 referenced by vb_ems_anotation/human_anotation_vb.csv),
# then run ems_whisper_eval_bundle/run_gap_analysis.py to quantify the
# DSP + Whisper-encoder gap between real VB audio and our v6 synth variants.
#
# Output → ems_whisper_eval_bundle/vb_gap_results/
#   gap_stats.csv       per-set DSP statistics
#   gap_distances.csv   Whisper encoder centroid/pairwise/MMD distances
#   gap_tsne.png        t-SNE plot
set -e

ROOT="/media/meow/One Touch/ems_call"
STAGE=/tmp/vb_real
OUT="${ROOT}/ems_whisper_eval_bundle/vb_gap_results"
PY=/home/meow/anaconda3/envs/asr_py10/bin/python

mkdir -p "${STAGE}"
rm -f "${STAGE}"/*.wav

cd "${ROOT}"
"${PY}" -c "
import pandas as pd, os
df = pd.read_csv('vb_ems_anotation/human_anotation_vb.csv')
df = df[df['transcript'].notna() & (df['transcript'].str.strip() != '')]
for f in df['Filename']:
    for d in ['random_samples_1','random_samples_2']:
        src = f'{d}/{f}'
        if os.path.exists(src):
            os.symlink(os.path.abspath(src), f'${STAGE}/{f}')
            break
n = len(os.listdir('${STAGE}'))
assert n == 50, f'expected 50 staged clips, got {n}'
print(f'staged {n} VB clips')
"

cd "${ROOT}/ems_whisper_eval_bundle"
rm -rf "${OUT}"
"${PY}" run_gap_analysis.py --real_wav_dir "${STAGE}" --n 50 --output_dir "${OUT}"
echo "==== Done. Results in ${OUT} ===="
