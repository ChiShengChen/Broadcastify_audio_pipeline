# CHiME4 Dataset Download Summary

## Status: Manual Download Required

The CHiME4 dataset could not be automatically downloaded due to expired download links. However, I've set up the directory structure and provided tools to help with manual download.

## What Was Accomplished

### 1. Directory Structure Created
✅ Created the complete CHiME4 directory structure at `/media/meow/One Touch/ems_call/CHiME4/`

```
CHiME4/
├── data/
│   ├── audio/
│   │   └── 16kHz/
│   │       ├── backgrounds/          # Background noise recordings
│   │       ├── embedded/            # Unsegmented noisy speech data
│   │       ├── isolated/            # Segmented noisy speech data
│   │       ├── isolated_1ch_track/  # 1-channel track data
│   │       ├── isolated_2ch_track/  # 2-channel track data
│   │       └── isolated_6ch_track/  # 6-channel track data
│   ├── annotations/                 # JSON annotation files
│   ├── transcriptions/             # Transcription files
│   └── WSJ0/                       # WSJ0 subset
```

### 2. Tools Created
✅ **download_chime4.py** - Automated download script (links expired)
✅ **download_chime4_manual.py** - Manual download helper and verification tool
✅ **CHiME4_download_guide.md** - Comprehensive download guide

## Download Options

### Option 1: LDC Package (Recommended)
- **URL**: https://catalog.ldc.upenn.edu/LDC2017S24
- **Cost**: Requires LDC license (paid)
- **Content**: Complete dataset with audio, annotations, and baseline software

### Option 2: Contact Organizers
- **Email**: chimechallenge@gmail.com
- **Request**: Access to CHiME4 dataset
- **Mention**: Whether you have a WSJ license

### Option 3: Academic Institution
- Check with your university's library or research computing center
- Many institutions have access to LDC datasets

## Dataset Information

### Content
- **Audio**: 16kHz WAV files from 4 environments (BUS, CAF, PED, STR)
- **Training**: 8,738 utterances (1,600 real + 7,138 simulated)
- **Development**: 3,280 utterances (410 real + 410 simulated per environment)
- **Test**: 2,640 utterances (330 real + 330 simulated per environment)

### Audio Specifications
- **Format**: 16-bit stereo WAV
- **Sample Rate**: 16 kHz
- **Channels**: 0-6 (0 = close-talk, 1-6 = tablet microphones)

## Next Steps

### 1. Choose Download Method
Select one of the download options above based on your situation:
- **Academic**: Try institutional access first
- **Research**: Contact organizers for access
- **Commercial**: Purchase LDC license

### 2. Download the Dataset
Follow the instructions in `CHiME4_download_guide.md`

### 3. Verify Download
After downloading, run:
```bash
python3 download_chime4_manual.py --verify
```

### 4. Analyze Dataset
Once downloaded, analyze the audio files:
```bash
python3 download_chime4_manual.py --analyze
```

## Files Created

1. **CHiME4/** - Main dataset directory with structure
2. **download_chime4.py** - Automated download script
3. **download_chime4_manual.py** - Manual download helper
4. **CHiME4_download_guide.md** - Comprehensive guide
5. **CHiME4_Summary.md** - This summary file

## Troubleshooting

### If Download Links Don't Work
- The official links have expired
- Contact organizers for updated links
- Try alternative sources mentioned in the guide

### If Dataset Structure is Incomplete
- Use the verification tool to identify missing components
- Ensure you have the complete LDC package
- Check file permissions and disk space

## Citation

When using CHiME4, cite:
```
Emmanuel Vincent, Shinji Watanabe, Aditya Arie Nugraha, Jon Barker, and Ricard Marxer
"An analysis of environment, microphone and data simulation mismatches in robust speech recognition"
Computer Speech and Language, 2017.
```

## Support

- **Email**: chimechallenge@gmail.com
- **Website**: https://www.chimechallenge.org/
- **Documentation**: See `CHiME4_download_guide.md`

---

**Note**: The CHiME4 dataset is primarily for research purposes. Commercial use may require additional licensing. 