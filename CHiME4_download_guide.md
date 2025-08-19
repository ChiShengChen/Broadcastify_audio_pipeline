# CHiME4 Dataset Download Guide

## Overview

The CHiME4 dataset is a challenging dataset for distant-talking automatic speech recognition. It contains real and simulated noisy speech data recorded in various environments (bus, cafe, pedestrian area, street junction).

## Dataset Structure

According to the [CHiME4 website](https://www.chimechallenge.org/challenges/chime4/data), the dataset includes:

```
CHiME4/data/
├── annotations/          # JSON annotation files
├── audio/               # Audio data (16kHz WAV files)
│   └── 16kHz/
│       ├── backgrounds/     # Background noise recordings
│       ├── embedded/        # Unsegmented noisy speech
│       ├── isolated/        # Segmented noisy speech
│       ├── isolated_1ch_track/
│       ├── isolated_2ch_track/
│       └── isolated_6ch_track/
├── transcriptions/      # Transcription files
└── WSJ0/               # WSJ0 subset
```

## Download Options

### 1. Official LDC Package (Recommended)

The main CHiME4 dataset is distributed via the Linguistic Data Consortium (LDC):
- **LDC Package**: [LDC2017S24](https://catalog.ldc.upenn.edu/LDC2017S24)
- **Requirements**: LDC license (paid)
- **Content**: Complete audio data, annotations, and baseline software

### 2. CHiME4_diff Package (Free)

A smaller package containing annotations and baseline code:
- **URL**: https://mab.to/dMwDNq4r2 (currently expired)
- **Content**: Annotations for 1ch/2ch tracks, cross-correlation coefficients, baseline code
- **License**: Apache 2.0

### 3. Alternative Sources

#### Option A: Contact CHiME Organizers
Email: chimechallenge@gmail.com
- Request access to the dataset
- Mention if you have a WSJ license

#### Option B: Academic Institutions
Many universities have access to LDC datasets. Check with your institution's library or research computing center.

#### Option C: Community Repositories
Some researchers have made portions of the dataset available:
- Check arXiv papers that use CHiME4
- Look for GitHub repositories that include sample data

## Manual Download Instructions

### Step 1: Get LDC Access
1. Visit [LDC Catalog](https://catalog.ldc.upenn.edu/LDC2017S24)
2. Purchase or obtain institutional access to LDC2017S24
3. Download the complete package

### Step 2: Download CHiME4_diff (if available)
1. Try the official link: https://mab.to/dMwDNq4r2
2. If expired, contact organizers for updated link
3. Extract the downloaded file

### Step 3: Verify Dataset Structure
After downloading, verify that you have:
- `data/audio/16kHz/isolated/` - Contains segmented audio files
- `data/annotations/` - Contains JSON annotation files
- `data/transcriptions/` - Contains transcription files

## Dataset Statistics

### Training Set
- 1600 real + 7138 simulated = 8738 utterances
- 4 speakers (real data)
- 83 speakers (simulated data)
- 4 environments: BUS, CAF, PED, STR

### Development Set
- 410 (real) × 4 (environments) + 410 (simulated) × 4 (environments) = 3280 utterances
- 4 speakers (different from training)

### Test Set
- 330 (real) × 4 (environments) + 330 (simulated) × 4 (environments) = 2640 utterances
- 4 speakers (different from training/development)

## Audio Specifications

- **Format**: 16-bit stereo WAV files
- **Sample Rate**: 16 kHz
- **Channels**: 0-6 (0 = close-talk microphone, 1-6 = tablet microphones)
- **Environments**: BTH (booth), BUS, CAF, PED, STR

## Citation

If you use CHiME4 in your research, please cite:

```
Emmanuel Vincent, Shinji Watanabe, Aditya Arie Nugraha, Jon Barker, and Ricard Marxer
"An analysis of environment, microphone and data simulation mismatches in robust speech recognition"
Computer Speech and Language, 2017.
```

## Troubleshooting

### Download Issues
- **Expired links**: Contact chimechallenge@gmail.com
- **LDC access**: Check with your institution's library
- **File corruption**: Verify checksums if provided

### Dataset Structure Issues
- Ensure all directories are present
- Check file permissions
- Verify audio file integrity

## Alternative Datasets

If CHiME4 is not available, consider these alternatives:
- **CHiME-5**: More recent, conversational speech
- **CHiME-6**: Multi-speaker conversations
- **Aurora-4**: Similar noisy speech recognition task
- **LibriSpeech**: Clean speech with noise simulation

## Support

For technical support or dataset access issues:
- Email: chimechallenge@gmail.com
- Website: https://www.chimechallenge.org/
- GitHub: https://github.com/chimechallenge/

## Notes

- The CHiME4 dataset is primarily for research purposes
- Commercial use may require additional licensing
- Some components may be available under different licenses
- Always check the latest information on the official website 