#!/bin/bash
set -e

# Auto-generated script to re-run ASR for missing files

# Create output directories
mkdir -p test_fix_results/audio_segments
mkdir -p test_fix_results/asr_transcripts
mkdir -p test_fix_results/merged_transcripts

echo '=== Processing large-v3 ==='
mkdir -p test_fix_results/audio_segments/large-v3
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_002.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_002.wav' 'test_fix_results/audio_segments/large-v3/long_audio_group_002.wav'
    echo 'Copied long_audio_group_002.wav for large-v3'
else
    echo 'Warning: Audio file long_audio_group_002.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_006.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_006.wav' 'test_fix_results/audio_segments/large-v3/long_audio_group_006.wav'
    echo 'Copied long_audio_group_006.wav for large-v3'
else
    echo 'Warning: Audio file long_audio_group_006.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_001.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_001.wav' 'test_fix_results/audio_segments/large-v3/long_audio_group_001.wav'
    echo 'Copied long_audio_group_001.wav for large-v3'
else
    echo 'Warning: Audio file long_audio_group_001.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_008.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_008.wav' 'test_fix_results/audio_segments/large-v3/long_audio_group_008.wav'
    echo 'Copied long_audio_group_008.wav for large-v3'
else
    echo 'Warning: Audio file long_audio_group_008.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_009.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_009.wav' 'test_fix_results/audio_segments/large-v3/long_audio_group_009.wav'
    echo 'Copied long_audio_group_009.wav for large-v3'
else
    echo 'Warning: Audio file long_audio_group_009.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_003.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_003.wav' 'test_fix_results/audio_segments/large-v3/long_audio_group_003.wav'
    echo 'Copied long_audio_group_003.wav for large-v3'
else
    echo 'Warning: Audio file long_audio_group_003.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_005.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_005.wav' 'test_fix_results/audio_segments/large-v3/long_audio_group_005.wav'
    echo 'Copied long_audio_group_005.wav for large-v3'
else
    echo 'Warning: Audio file long_audio_group_005.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_004.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_004.wav' 'test_fix_results/audio_segments/large-v3/long_audio_group_004.wav'
    echo 'Copied long_audio_group_004.wav for large-v3'
else
    echo 'Warning: Audio file long_audio_group_004.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_007.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_007.wav' 'test_fix_results/audio_segments/large-v3/long_audio_group_007.wav'
    echo 'Copied long_audio_group_007.wav for large-v3'
else
    echo 'Warning: Audio file long_audio_group_007.wav not found'
fi

echo 'Running ASR for large-v3...'
python3 run_all_asrs.py test_fix_results/audio_segments/large-v3

echo 'Moving transcripts for large-v3...'
if [ -f 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_002.txt' ]; then
    mv 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_002.txt' 'test_fix_results/asr_transcripts/large-v3_long_audio_group_002.txt'
    echo 'Moved large-v3_long_audio_group_002.txt'
else
    echo 'Warning: Transcript large-v3_long_audio_group_002.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_006.txt' ]; then
    mv 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_006.txt' 'test_fix_results/asr_transcripts/large-v3_long_audio_group_006.txt'
    echo 'Moved large-v3_long_audio_group_006.txt'
else
    echo 'Warning: Transcript large-v3_long_audio_group_006.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_001.txt' ]; then
    mv 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_001.txt' 'test_fix_results/asr_transcripts/large-v3_long_audio_group_001.txt'
    echo 'Moved large-v3_long_audio_group_001.txt'
else
    echo 'Warning: Transcript large-v3_long_audio_group_001.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_008.txt' ]; then
    mv 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_008.txt' 'test_fix_results/asr_transcripts/large-v3_long_audio_group_008.txt'
    echo 'Moved large-v3_long_audio_group_008.txt'
else
    echo 'Warning: Transcript large-v3_long_audio_group_008.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_009.txt' ]; then
    mv 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_009.txt' 'test_fix_results/asr_transcripts/large-v3_long_audio_group_009.txt'
    echo 'Moved large-v3_long_audio_group_009.txt'
else
    echo 'Warning: Transcript large-v3_long_audio_group_009.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_003.txt' ]; then
    mv 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_003.txt' 'test_fix_results/asr_transcripts/large-v3_long_audio_group_003.txt'
    echo 'Moved large-v3_long_audio_group_003.txt'
else
    echo 'Warning: Transcript large-v3_long_audio_group_003.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_005.txt' ]; then
    mv 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_005.txt' 'test_fix_results/asr_transcripts/large-v3_long_audio_group_005.txt'
    echo 'Moved large-v3_long_audio_group_005.txt'
else
    echo 'Warning: Transcript large-v3_long_audio_group_005.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_004.txt' ]; then
    mv 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_004.txt' 'test_fix_results/asr_transcripts/large-v3_long_audio_group_004.txt'
    echo 'Moved large-v3_long_audio_group_004.txt'
else
    echo 'Warning: Transcript large-v3_long_audio_group_004.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_007.txt' ]; then
    mv 'test_fix_results/audio_segments/large-v3/large-v3_long_audio_group_007.txt' 'test_fix_results/asr_transcripts/large-v3_long_audio_group_007.txt'
    echo 'Moved large-v3_long_audio_group_007.txt'
else
    echo 'Warning: Transcript large-v3_long_audio_group_007.txt not generated'
fi

echo '=== Processing canary-1b ==='
mkdir -p test_fix_results/audio_segments/canary-1b
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_002.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_002.wav' 'test_fix_results/audio_segments/canary-1b/long_audio_group_002.wav'
    echo 'Copied long_audio_group_002.wav for canary-1b'
else
    echo 'Warning: Audio file long_audio_group_002.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_006.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_006.wav' 'test_fix_results/audio_segments/canary-1b/long_audio_group_006.wav'
    echo 'Copied long_audio_group_006.wav for canary-1b'
else
    echo 'Warning: Audio file long_audio_group_006.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_001.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_001.wav' 'test_fix_results/audio_segments/canary-1b/long_audio_group_001.wav'
    echo 'Copied long_audio_group_001.wav for canary-1b'
else
    echo 'Warning: Audio file long_audio_group_001.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_008.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_008.wav' 'test_fix_results/audio_segments/canary-1b/long_audio_group_008.wav'
    echo 'Copied long_audio_group_008.wav for canary-1b'
else
    echo 'Warning: Audio file long_audio_group_008.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_009.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_009.wav' 'test_fix_results/audio_segments/canary-1b/long_audio_group_009.wav'
    echo 'Copied long_audio_group_009.wav for canary-1b'
else
    echo 'Warning: Audio file long_audio_group_009.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_003.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_003.wav' 'test_fix_results/audio_segments/canary-1b/long_audio_group_003.wav'
    echo 'Copied long_audio_group_003.wav for canary-1b'
else
    echo 'Warning: Audio file long_audio_group_003.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_005.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_005.wav' 'test_fix_results/audio_segments/canary-1b/long_audio_group_005.wav'
    echo 'Copied long_audio_group_005.wav for canary-1b'
else
    echo 'Warning: Audio file long_audio_group_005.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_004.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_004.wav' 'test_fix_results/audio_segments/canary-1b/long_audio_group_004.wav'
    echo 'Copied long_audio_group_004.wav for canary-1b'
else
    echo 'Warning: Audio file long_audio_group_004.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_007.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_007.wav' 'test_fix_results/audio_segments/canary-1b/long_audio_group_007.wav'
    echo 'Copied long_audio_group_007.wav for canary-1b'
else
    echo 'Warning: Audio file long_audio_group_007.wav not found'
fi

echo 'Running ASR for canary-1b...'
python3 run_all_asrs.py test_fix_results/audio_segments/canary-1b

echo 'Moving transcripts for canary-1b...'
if [ -f 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_002.txt' ]; then
    mv 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_002.txt' 'test_fix_results/asr_transcripts/canary-1b_long_audio_group_002.txt'
    echo 'Moved canary-1b_long_audio_group_002.txt'
else
    echo 'Warning: Transcript canary-1b_long_audio_group_002.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_006.txt' ]; then
    mv 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_006.txt' 'test_fix_results/asr_transcripts/canary-1b_long_audio_group_006.txt'
    echo 'Moved canary-1b_long_audio_group_006.txt'
else
    echo 'Warning: Transcript canary-1b_long_audio_group_006.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_001.txt' ]; then
    mv 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_001.txt' 'test_fix_results/asr_transcripts/canary-1b_long_audio_group_001.txt'
    echo 'Moved canary-1b_long_audio_group_001.txt'
else
    echo 'Warning: Transcript canary-1b_long_audio_group_001.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_008.txt' ]; then
    mv 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_008.txt' 'test_fix_results/asr_transcripts/canary-1b_long_audio_group_008.txt'
    echo 'Moved canary-1b_long_audio_group_008.txt'
else
    echo 'Warning: Transcript canary-1b_long_audio_group_008.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_009.txt' ]; then
    mv 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_009.txt' 'test_fix_results/asr_transcripts/canary-1b_long_audio_group_009.txt'
    echo 'Moved canary-1b_long_audio_group_009.txt'
else
    echo 'Warning: Transcript canary-1b_long_audio_group_009.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_003.txt' ]; then
    mv 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_003.txt' 'test_fix_results/asr_transcripts/canary-1b_long_audio_group_003.txt'
    echo 'Moved canary-1b_long_audio_group_003.txt'
else
    echo 'Warning: Transcript canary-1b_long_audio_group_003.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_005.txt' ]; then
    mv 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_005.txt' 'test_fix_results/asr_transcripts/canary-1b_long_audio_group_005.txt'
    echo 'Moved canary-1b_long_audio_group_005.txt'
else
    echo 'Warning: Transcript canary-1b_long_audio_group_005.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_004.txt' ]; then
    mv 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_004.txt' 'test_fix_results/asr_transcripts/canary-1b_long_audio_group_004.txt'
    echo 'Moved canary-1b_long_audio_group_004.txt'
else
    echo 'Warning: Transcript canary-1b_long_audio_group_004.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_007.txt' ]; then
    mv 'test_fix_results/audio_segments/canary-1b/canary-1b_long_audio_group_007.txt' 'test_fix_results/asr_transcripts/canary-1b_long_audio_group_007.txt'
    echo 'Moved canary-1b_long_audio_group_007.txt'
else
    echo 'Warning: Transcript canary-1b_long_audio_group_007.txt not generated'
fi

echo '=== Processing parakeet-tdt-0.6b-v2 ==='
mkdir -p test_fix_results/audio_segments/parakeet-tdt-0.6b-v2
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_002.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_002.wav' 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/long_audio_group_002.wav'
    echo 'Copied long_audio_group_002.wav for parakeet-tdt-0.6b-v2'
else
    echo 'Warning: Audio file long_audio_group_002.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_006.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_006.wav' 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/long_audio_group_006.wav'
    echo 'Copied long_audio_group_006.wav for parakeet-tdt-0.6b-v2'
else
    echo 'Warning: Audio file long_audio_group_006.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_001.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_001.wav' 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/long_audio_group_001.wav'
    echo 'Copied long_audio_group_001.wav for parakeet-tdt-0.6b-v2'
else
    echo 'Warning: Audio file long_audio_group_001.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_008.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_008.wav' 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/long_audio_group_008.wav'
    echo 'Copied long_audio_group_008.wav for parakeet-tdt-0.6b-v2'
else
    echo 'Warning: Audio file long_audio_group_008.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_009.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_009.wav' 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/long_audio_group_009.wav'
    echo 'Copied long_audio_group_009.wav for parakeet-tdt-0.6b-v2'
else
    echo 'Warning: Audio file long_audio_group_009.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_003.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_003.wav' 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/long_audio_group_003.wav'
    echo 'Copied long_audio_group_003.wav for parakeet-tdt-0.6b-v2'
else
    echo 'Warning: Audio file long_audio_group_003.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_005.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_005.wav' 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/long_audio_group_005.wav'
    echo 'Copied long_audio_group_005.wav for parakeet-tdt-0.6b-v2'
else
    echo 'Warning: Audio file long_audio_group_005.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_004.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_004.wav' 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/long_audio_group_004.wav'
    echo 'Copied long_audio_group_004.wav for parakeet-tdt-0.6b-v2'
else
    echo 'Warning: Audio file long_audio_group_004.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_007.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_007.wav' 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/long_audio_group_007.wav'
    echo 'Copied long_audio_group_007.wav for parakeet-tdt-0.6b-v2'
else
    echo 'Warning: Audio file long_audio_group_007.wav not found'
fi

echo 'Running ASR for parakeet-tdt-0.6b-v2...'
python3 run_all_asrs.py test_fix_results/audio_segments/parakeet-tdt-0.6b-v2

echo 'Moving transcripts for parakeet-tdt-0.6b-v2...'
if [ -f 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_002.txt' ]; then
    mv 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_002.txt' 'test_fix_results/asr_transcripts/parakeet-tdt-0.6b-v2_long_audio_group_002.txt'
    echo 'Moved parakeet-tdt-0.6b-v2_long_audio_group_002.txt'
else
    echo 'Warning: Transcript parakeet-tdt-0.6b-v2_long_audio_group_002.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_006.txt' ]; then
    mv 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_006.txt' 'test_fix_results/asr_transcripts/parakeet-tdt-0.6b-v2_long_audio_group_006.txt'
    echo 'Moved parakeet-tdt-0.6b-v2_long_audio_group_006.txt'
else
    echo 'Warning: Transcript parakeet-tdt-0.6b-v2_long_audio_group_006.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_001.txt' ]; then
    mv 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_001.txt' 'test_fix_results/asr_transcripts/parakeet-tdt-0.6b-v2_long_audio_group_001.txt'
    echo 'Moved parakeet-tdt-0.6b-v2_long_audio_group_001.txt'
else
    echo 'Warning: Transcript parakeet-tdt-0.6b-v2_long_audio_group_001.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_008.txt' ]; then
    mv 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_008.txt' 'test_fix_results/asr_transcripts/parakeet-tdt-0.6b-v2_long_audio_group_008.txt'
    echo 'Moved parakeet-tdt-0.6b-v2_long_audio_group_008.txt'
else
    echo 'Warning: Transcript parakeet-tdt-0.6b-v2_long_audio_group_008.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_009.txt' ]; then
    mv 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_009.txt' 'test_fix_results/asr_transcripts/parakeet-tdt-0.6b-v2_long_audio_group_009.txt'
    echo 'Moved parakeet-tdt-0.6b-v2_long_audio_group_009.txt'
else
    echo 'Warning: Transcript parakeet-tdt-0.6b-v2_long_audio_group_009.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_003.txt' ]; then
    mv 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_003.txt' 'test_fix_results/asr_transcripts/parakeet-tdt-0.6b-v2_long_audio_group_003.txt'
    echo 'Moved parakeet-tdt-0.6b-v2_long_audio_group_003.txt'
else
    echo 'Warning: Transcript parakeet-tdt-0.6b-v2_long_audio_group_003.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_005.txt' ]; then
    mv 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_005.txt' 'test_fix_results/asr_transcripts/parakeet-tdt-0.6b-v2_long_audio_group_005.txt'
    echo 'Moved parakeet-tdt-0.6b-v2_long_audio_group_005.txt'
else
    echo 'Warning: Transcript parakeet-tdt-0.6b-v2_long_audio_group_005.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_004.txt' ]; then
    mv 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_004.txt' 'test_fix_results/asr_transcripts/parakeet-tdt-0.6b-v2_long_audio_group_004.txt'
    echo 'Moved parakeet-tdt-0.6b-v2_long_audio_group_004.txt'
else
    echo 'Warning: Transcript parakeet-tdt-0.6b-v2_long_audio_group_004.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_007.txt' ]; then
    mv 'test_fix_results/audio_segments/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2_long_audio_group_007.txt' 'test_fix_results/asr_transcripts/parakeet-tdt-0.6b-v2_long_audio_group_007.txt'
    echo 'Moved parakeet-tdt-0.6b-v2_long_audio_group_007.txt'
else
    echo 'Warning: Transcript parakeet-tdt-0.6b-v2_long_audio_group_007.txt not generated'
fi

echo '=== Processing wav2vec-xls-r ==='
mkdir -p test_fix_results/audio_segments/wav2vec-xls-r
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_002.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_002.wav' 'test_fix_results/audio_segments/wav2vec-xls-r/long_audio_group_002.wav'
    echo 'Copied long_audio_group_002.wav for wav2vec-xls-r'
else
    echo 'Warning: Audio file long_audio_group_002.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_006.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_006.wav' 'test_fix_results/audio_segments/wav2vec-xls-r/long_audio_group_006.wav'
    echo 'Copied long_audio_group_006.wav for wav2vec-xls-r'
else
    echo 'Warning: Audio file long_audio_group_006.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_001.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_001.wav' 'test_fix_results/audio_segments/wav2vec-xls-r/long_audio_group_001.wav'
    echo 'Copied long_audio_group_001.wav for wav2vec-xls-r'
else
    echo 'Warning: Audio file long_audio_group_001.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_008.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_008.wav' 'test_fix_results/audio_segments/wav2vec-xls-r/long_audio_group_008.wav'
    echo 'Copied long_audio_group_008.wav for wav2vec-xls-r'
else
    echo 'Warning: Audio file long_audio_group_008.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_009.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_009.wav' 'test_fix_results/audio_segments/wav2vec-xls-r/long_audio_group_009.wav'
    echo 'Copied long_audio_group_009.wav for wav2vec-xls-r'
else
    echo 'Warning: Audio file long_audio_group_009.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_003.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_003.wav' 'test_fix_results/audio_segments/wav2vec-xls-r/long_audio_group_003.wav'
    echo 'Copied long_audio_group_003.wav for wav2vec-xls-r'
else
    echo 'Warning: Audio file long_audio_group_003.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_005.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_005.wav' 'test_fix_results/audio_segments/wav2vec-xls-r/long_audio_group_005.wav'
    echo 'Copied long_audio_group_005.wav for wav2vec-xls-r'
else
    echo 'Warning: Audio file long_audio_group_005.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_004.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_004.wav' 'test_fix_results/audio_segments/wav2vec-xls-r/long_audio_group_004.wav'
    echo 'Copied long_audio_group_004.wav for wav2vec-xls-r'
else
    echo 'Warning: Audio file long_audio_group_004.wav not found'
fi
if [ -f '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_007.wav' ]; then
    cp '$(dirname 'long_audio_test_dataset/long_audio_ground_truth.csv')/long_audio_group_007.wav' 'test_fix_results/audio_segments/wav2vec-xls-r/long_audio_group_007.wav'
    echo 'Copied long_audio_group_007.wav for wav2vec-xls-r'
else
    echo 'Warning: Audio file long_audio_group_007.wav not found'
fi

echo 'Running ASR for wav2vec-xls-r...'
python3 run_all_asrs.py test_fix_results/audio_segments/wav2vec-xls-r

echo 'Moving transcripts for wav2vec-xls-r...'
if [ -f 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_002.txt' ]; then
    mv 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_002.txt' 'test_fix_results/asr_transcripts/wav2vec-xls-r_long_audio_group_002.txt'
    echo 'Moved wav2vec-xls-r_long_audio_group_002.txt'
else
    echo 'Warning: Transcript wav2vec-xls-r_long_audio_group_002.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_006.txt' ]; then
    mv 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_006.txt' 'test_fix_results/asr_transcripts/wav2vec-xls-r_long_audio_group_006.txt'
    echo 'Moved wav2vec-xls-r_long_audio_group_006.txt'
else
    echo 'Warning: Transcript wav2vec-xls-r_long_audio_group_006.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_001.txt' ]; then
    mv 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_001.txt' 'test_fix_results/asr_transcripts/wav2vec-xls-r_long_audio_group_001.txt'
    echo 'Moved wav2vec-xls-r_long_audio_group_001.txt'
else
    echo 'Warning: Transcript wav2vec-xls-r_long_audio_group_001.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_008.txt' ]; then
    mv 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_008.txt' 'test_fix_results/asr_transcripts/wav2vec-xls-r_long_audio_group_008.txt'
    echo 'Moved wav2vec-xls-r_long_audio_group_008.txt'
else
    echo 'Warning: Transcript wav2vec-xls-r_long_audio_group_008.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_009.txt' ]; then
    mv 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_009.txt' 'test_fix_results/asr_transcripts/wav2vec-xls-r_long_audio_group_009.txt'
    echo 'Moved wav2vec-xls-r_long_audio_group_009.txt'
else
    echo 'Warning: Transcript wav2vec-xls-r_long_audio_group_009.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_003.txt' ]; then
    mv 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_003.txt' 'test_fix_results/asr_transcripts/wav2vec-xls-r_long_audio_group_003.txt'
    echo 'Moved wav2vec-xls-r_long_audio_group_003.txt'
else
    echo 'Warning: Transcript wav2vec-xls-r_long_audio_group_003.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_005.txt' ]; then
    mv 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_005.txt' 'test_fix_results/asr_transcripts/wav2vec-xls-r_long_audio_group_005.txt'
    echo 'Moved wav2vec-xls-r_long_audio_group_005.txt'
else
    echo 'Warning: Transcript wav2vec-xls-r_long_audio_group_005.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_004.txt' ]; then
    mv 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_004.txt' 'test_fix_results/asr_transcripts/wav2vec-xls-r_long_audio_group_004.txt'
    echo 'Moved wav2vec-xls-r_long_audio_group_004.txt'
else
    echo 'Warning: Transcript wav2vec-xls-r_long_audio_group_004.txt not generated'
fi
if [ -f 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_007.txt' ]; then
    mv 'test_fix_results/audio_segments/wav2vec-xls-r/wav2vec-xls-r_long_audio_group_007.txt' 'test_fix_results/asr_transcripts/wav2vec-xls-r_long_audio_group_007.txt'
    echo 'Moved wav2vec-xls-r_long_audio_group_007.txt'
else
    echo 'Warning: Transcript wav2vec-xls-r_long_audio_group_007.txt not generated'
fi

echo '=== Copying existing transcripts for reference ==='
if [ -d 'pipeline_csside_results_20250726_070330/merged_transcripts' ]; then
    cp -r 'pipeline_csside_results_20250726_070330/merged_transcripts'/* 'test_fix_results/merged_transcripts/' 2>/dev/null || true
    echo 'Copied existing transcripts'
fi

echo '=== Merging transcripts ==='
if [ -d 'pipeline_csside_results_20250726_070330/long_audio_segments' ]; then
    python3 merge_split_transcripts.py \
        --input_dir 'test_fix_results/asr_transcripts' \
        --output_dir 'test_fix_results/merged_transcripts' \
        --metadata_dir 'pipeline_csside_results_20250726_070330/long_audio_segments'
    echo 'Transcript merging completed'
fi

echo '=== Running evaluation ==='
python3 evaluate_asr.py \
    --transcript_dirs 'test_fix_results/merged_transcripts' \
    --ground_truth_file 'long_audio_test_dataset/long_audio_ground_truth.csv' \
    --output_file 'test_fix_results/asr_evaluation_results_fixed.csv'

echo '=== Running model file analysis ==='
python3 analyze_model_files.py \
    --transcript_dir 'test_fix_results/merged_transcripts' \
    --ground_truth_file 'long_audio_test_dataset/long_audio_ground_truth.csv' \
    --output_file 'test_fix_results/model_file_analysis_fixed.txt'

echo '=== Re-run completed ==='