# CHiME4 Download Debug Report

## Issue Identified

**Problem**: The CHiME4 dataset download failed because the official download links have expired.

**Root Cause**: The download link `https://mab.to/dMwDNq4r2` redirects to MyAirBridge.com, which is a file sharing service that requires authentication or has expired links.

## What Happened

1. **Download Attempt**: The script tried to download from the official CHiME4 link
2. **Redirect**: The link redirected to MyAirBridge.com instead of providing the actual dataset
3. **HTML Response**: Instead of a zip file, we received an HTML page from the file sharing service
4. **Extraction Failure**: The "zip" file was actually HTML content, causing the unzip command to fail

## Error Analysis

### Terminal Output Analysis
```bash
unzip CHiME4_diff_v1.0.zip 
Archive:  CHiME4_diff_v1.0.zip
  End-of-central-directory signature not found.  Either this file is not
  a zipfile, or it constitutes one disk of a multi-part archive.
```

**Translation**: The file is not a valid zip archive because it contains HTML content, not compressed data.

### File Type Verification
```bash
file CHiME4/CHiME4_diff_v1.0.zip
# Output: HTML document, ASCII text, with very long lines
```

**Translation**: The file is an HTML webpage, not a zip archive.

## Current Status

### ✅ What's Working
- Directory structure is properly created
- Verification tools are functional
- Documentation is complete

### ❌ What's Missing
- Actual CHiME4 dataset files
- Audio data (WAV files)
- Annotation files (JSON)
- Transcription files
- WSJ0 subset

## Solutions

### Option 1: LDC Package (Recommended)
```bash
# Visit this URL and purchase/obtain access
https://catalog.ldc.upenn.edu/LDC2017S24
```

### Option 2: Contact Organizers
```bash
# Email the CHiME organizers
chimechallenge@gmail.com
```

### Option 3: Academic Institution Access
- Check with your university's library
- Many institutions have LDC access

## Verification Commands

### Check Current Status
```bash
python3 download_chime4_manual.py --verify
```

### Show Download Instructions
```bash
python3 download_chime4_manual.py --instructions
```

### Analyze Audio Files (when available)
```bash
python3 download_chime4_manual.py --analyze
```

## Expected Dataset Structure

Once properly downloaded, you should have:

```
CHiME4/
├── data/
│   ├── audio/16kHz/
│   │   ├── backgrounds/          # 102 WAV files
│   │   ├── embedded/            # 357 WAV files  
│   │   ├── isolated/            # ~50,000+ WAV files
│   │   ├── isolated_1ch_track/  # Subset for 1-channel
│   │   ├── isolated_2ch_track/  # Subset for 2-channel
│   │   └── isolated_6ch_track/  # All channels
│   ├── annotations/
│   │   ├── dt05_real.json       # Development real annotations
│   │   ├── dt05_simu.json       # Development simulated annotations
│   │   ├── tr05_simu.json       # Training simulated annotations
│   │   └── mic_error.csv        # Microphone error data
│   ├── transcriptions/
│   │   ├── dt05_real.dot_all    # Development real transcriptions
│   │   └── dt05_simu.dot_all    # Development simulated transcriptions
│   └── WSJ0/                    # WSJ0 subset
```

## Dataset Statistics

When complete, you should have:
- **Training**: 8,738 utterances
- **Development**: 3,280 utterances  
- **Test**: 2,640 utterances
- **Total Audio**: ~50,000+ WAV files
- **Size**: Several GB of audio data

## Next Steps

1. **Choose download method** from the options above
2. **Download the dataset** using your chosen method
3. **Verify the download** using the verification tool
4. **Analyze the data** to ensure everything is correct

## Troubleshooting Commands

### Check Disk Space
```bash
df -h /media/meow/One\ Touch/ems_call/CHiME4/
```

### Check File Permissions
```bash
ls -la CHiME4/data/
```

### Count Files (when downloaded)
```bash
find CHiME4/data/audio/16kHz/isolated -name "*.wav" | wc -l
```

## Support Resources

- **Email**: chimechallenge@gmail.com
- **Website**: https://www.chimechallenge.org/
- **Documentation**: See `CHiME4_download_guide.md`

---

**Status**: Ready for manual download. All tools and structure are in place. 