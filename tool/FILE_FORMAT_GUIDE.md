# File Format Guide

A comprehensive guide to input and output file specifications for the EMS Call ASR and LLM-Enhanced Pipeline.

## 📋 Overview

This guide details all file formats, structures, and specifications used throughout both pipeline stages, including input requirements, output formats, and intermediate file structures.

## 📥 Input File Formats

### Audio Files (Stage 1 Input)

#### Supported Audio Formats

| Format | Extension | Recommended | Notes |
|--------|-----------|-------------|-------|
| **WAV** | `.wav` | ✅ **Yes** | Preferred format, no compression |
| **MP3** | `.mp3` | ⚠️ Converted | Auto-converted to WAV |
| **FLAC** | `.flac` | ⚠️ Converted | Lossless, auto-converted |
| **M4A** | `.m4a` | ⚠️ Converted | Auto-converted to WAV |

#### Audio Specifications

| Parameter | Recommended | Supported Range | Auto-Conversion |
|-----------|-------------|-----------------|-----------------|
| **Sample Rate** | 16 kHz | 8-48 kHz | Yes, to 16 kHz |
| **Channels** | Mono (1) | Mono/Stereo | Yes, to Mono |
| **Bit Depth** | 16-bit | 16/24/32-bit | Yes, to 16-bit |
| **Duration** | Any | 1 second - 8 hours | Auto-split if >2 hours |

#### Audio File Naming

```
# Recommended naming convention
call_001.wav
call_002.wav
emergency_20240101_001.wav
ems_dispatch_001.wav

# Avoid special characters
✅ Good: call_001.wav, emergency-call-001.wav
❌ Bad: call#001.wav, emergency call 001.wav
```

### Ground Truth Files (Both Stages)

#### CSV Format Specification

**Required Structure:**
```csv
Filename,transcript
call_001.wav,"Patient reports chest pain and shortness of breath"
call_002.wav,"Motor vehicle accident at Main Street intersection"
call_003.wav,"Elderly patient fell at home, possible hip fracture"
```

#### CSV Requirements

| Element | Requirement | Description |
|---------|-------------|-------------|
| **Header** | Required | Must be exactly: `Filename,transcript` |
| **Encoding** | UTF-8 | Unicode support for special characters |
| **Delimiter** | Comma (`,`) | Standard CSV delimiter |
| **Quoting** | Double quotes (`"`) | For text containing commas |
| **Line Endings** | Unix (`\n`) or Windows (`\r\n`) | Both supported |

#### Filename Column Requirements

```csv
# ✅ Correct filename matching
Filename,transcript
call_001.wav,"Transcript content here"

# ❌ Common mistakes
call_001,"Missing file extension"
Call_001.wav,"Case mismatch with actual file"
call_001.WAV,"Extension case mismatch"
```

#### Transcript Column Requirements

| Aspect | Requirement | Example |
|--------|-------------|---------|
| **Language** | English (primary) | "Patient has chest pain" |
| **Punctuation** | Optional | "Patient has chest pain." or "Patient has chest pain" |
| **Case** | Any | "Patient has chest pain" or "PATIENT HAS CHEST PAIN" |
| **Medical Terms** | As spoken | "Patient has MI" not "Patient has myocardial infarction" |
| **Numbers** | As spoken | "Give 325 milligrams" not "Give three hundred twenty-five mg" |

#### Example Ground Truth Files

**Basic Medical Calls:**
```csv
Filename,transcript
cardiac_001.wav,"67-year-old male with chest pain radiating to left arm"
respiratory_002.wav,"Female patient having difficulty breathing and wheezing"
trauma_003.wav,"Motor vehicle collision with possible head injury"
pediatric_004.wav,"3-year-old with high fever and difficulty swallowing"
```

**With Medical Terminology:**
```csv
Filename,transcript
call_001.wav,"Patient presents with acute MI, administered aspirin 325mg"
call_002.wav,"Respiratory distress, O2 sat 88%, started on albuterol"
call_003.wav,"Trauma alert, GCS 12, possible C-spine injury"
call_004.wav,"Pediatric fever, temp 104F, possible febrile seizure"
```

## 📤 Output File Formats

### Stage 1: ASR Pipeline Outputs

#### Directory Structure
```
pipeline_results_YYYYMMDD_HHMMSS/
├── preprocessed_audio/              # Audio preprocessing results
├── filtered_audio/                  # Audio filtering results
├── vad_segments/                    # VAD extracted segments
├── long_audio_segments/             # Long audio split segments
├── asr_transcripts/                 # Raw ASR transcription results
├── merged_transcripts/              # Merged segmented transcripts
├── asr_evaluation_results.csv       # Performance metrics
├── model_file_analysis.txt          # Model processing analysis
├── error_analysis.log               # Error tracking
└── pipeline_summary.txt             # Complete processing summary
```

#### ASR Transcript Files

**File Naming Convention:**
```
# Format: [model]_[original_filename].txt
large-v3_call_001.txt               # Whisper Large-v3 result
wav2vec2_call_001.txt               # Wav2Vec2 result
canary-1b_call_001.txt              # Canary-1B result
parakeet_call_001.txt               # Parakeet result
```

**Content Format:**
```
# File: large-v3_call_001.txt
Patient reports chest pain and shortness of breath. Pain started about 30 minutes ago. Patient is a 65-year-old male with history of hypertension.
```

#### Evaluation Results CSV

**Structure:**
```csv
model,filename,wer,mer,wil,cer,processing_time_seconds,file_size_bytes,transcript_length,ground_truth_length
large-v3,call_001.wav,0.15,0.12,0.18,0.08,2.34,1024,156,142
wav2vec2,call_001.wav,0.22,0.19,0.25,0.14,1.87,1024,148,142
canary-1b,call_001.wav,0.18,0.15,0.21,0.11,3.12,1024,151,142
```

**Column Descriptions:**
- `model`: ASR model name
- `filename`: Original audio filename
- `wer`: Word Error Rate (0-1, lower is better)
- `mer`: Match Error Rate (0-1, lower is better)
- `wil`: Word Information Lost (0-1, lower is better)
- `cer`: Character Error Rate (0-1, lower is better)
- `processing_time_seconds`: Processing duration
- `file_size_bytes`: Original audio file size
- `transcript_length`: Generated transcript word count
- `ground_truth_length`: Ground truth word count

### Stage 2: LLM Pipeline Outputs

#### Directory Structure
```
llm_results_YYYYMMDD_HHMMSS/
├── whisper_filtered/                    # Filtered Whisper transcripts
├── corrected_transcripts/               # Medical term corrected transcripts
├── emergency_pages/                     # Generated emergency pages
├── llm_enhanced_evaluation_results.csv # Enhanced evaluation metrics
├── error_analysis.log                  # Detailed error tracking
└── llm_enhanced_pipeline_summary.txt   # Processing summary
```

#### Medical Correction Results

**File Format:**
```
# File: corrected_transcripts/large-v3_call_001.txt
Patient reports chest pain and dyspnea. Pain started approximately 30 minutes ago. Patient is a 65-year-old male with history of hypertension.
```

**Summary JSON:**
```json
{
  "processing_summary": {
    "total_files": 150,
    "successful_corrections": 148,
    "failed_corrections": 2,
    "processing_time": "00:45:23",
    "model_used": "BioMistral-7B",
    "quantization": "8-bit"
  },
  "correction_statistics": {
    "medical_terms_corrected": 342,
    "drug_names_standardized": 89,
    "anatomical_terms_corrected": 156,
    "procedure_names_standardized": 97
  },
  "failed_files": [
    {
      "file": "large-v3_call_010.txt",
      "error": "Empty or unreadable transcript",
      "timestamp": "2025-08-13 07:51:40"
    }
  ]
}
```

#### Emergency Page Results

**File Format:**
```
# File: emergency_pages/large-v3_call_001_emergency_page.txt
=== EMERGENCY MEDICAL DISPATCH ===
PRIORITY LEVEL: HIGH
INCIDENT TYPE: Cardiac Emergency

PATIENT INFORMATION:
- Age: 65 years old
- Gender: Male
- Chief Complaint: Chest pain with dyspnea

CLINICAL PRESENTATION:
- Chest pain radiating to left arm
- Shortness of breath
- Onset: 30 minutes ago
- Medical History: Hypertension

LOCATION DETAILS:
- Address: [To be determined from call]
- Access: [Standard residential access]

RESOURCES REQUESTED:
- ALS Ambulance
- Cardiac monitor
- IV access capability
- Consider ALS intercept if BLS first response

TRANSPORT PRIORITY:
- Destination: Nearest PCI-capable facility
- ETA: Critical transport
- Notify receiving facility of possible STEMI

SPECIAL CONSIDERATIONS:
- Monitor for cardiac arrest
- Prepare for possible CPR/defibrillation
- Continuous cardiac monitoring required
```

## 📊 Intermediate File Formats

### Processing Metadata

#### VAD Segment Information
```json
{
  "original_file": "call_001.wav",
  "segments": [
    {
      "segment_id": 1,
      "start_time": 0.0,
      "end_time": 12.5,
      "duration": 12.5,
      "confidence": 0.89,
      "output_file": "call_001_segment_001.wav"
    },
    {
      "segment_id": 2,
      "start_time": 15.2,
      "end_time": 28.7,
      "duration": 13.5,
      "confidence": 0.92,
      "output_file": "call_001_segment_002.wav"
    }
  ],
  "total_speech_duration": 26.0,
  "total_silence_removed": 34.8,
  "processing_time": 1.23
}
```

#### Long Audio Splitting Information
```json
{
  "original_file": "long_call_001.wav",
  "original_duration": 1800.0,
  "max_segment_duration": 120.0,
  "segments": [
    {
      "segment_number": 1,
      "start_time": 0.0,
      "end_time": 120.0,
      "output_file": "long_call_001_segment_001.wav",
      "boundary_type": "silence_detection"
    },
    {
      "segment_number": 2,
      "start_time": 120.0,
      "end_time": 240.0,
      "output_file": "long_call_001_segment_002.wav",
      "boundary_type": "speech_pause"
    }
  ],
  "total_segments": 15,
  "processing_method": "intelligent_boundary_detection"
}
```

## 🔧 Configuration File Formats

### VAD Configuration (JSON)
```json
{
  "vad_config": {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": 30,
    "speech_pad_ms": 30,
    "frame_length_ms": 30,
    "hop_length_ms": 10
  },
  "audio_config": {
    "sample_rate": 16000,
    "channels": 1,
    "bit_depth": 16
  }
}
```

### Model Configuration (JSON)
```json
{
  "asr_models": {
    "large-v3": {
      "framework": "whisper",
      "model_path": "large-v3",
      "enabled": true,
      "config": {
        "language": "en",
        "task": "transcribe",
        "temperature": 0.0
      }
    }
  },
  "llm_models": {
    "BioMistral-7B": {
      "framework": "transformers",
      "model_path": "BioMistral/BioMistral-7B",
      "quantization": {
        "load_in_8bit": true,
        "load_in_4bit": false
      },
      "generation_config": {
        "max_length": 512,
        "temperature": 0.1,
        "do_sample": false
      }
    }
  }
}
```

## 📝 Log File Formats

### Error Analysis Log
```
=== LLM-Enhanced Pipeline Error Analysis Log ===
Analysis Date: 2025-08-13 07:47:01 CST
Pipeline Output Directory: /path/to/llm_results_20250813_074700
ASR Results Directory: /path/to/pipeline_results_20250729_033902

FAILED FILE: /path/to/large-v3_call_010.txt
  Processing Mode: medical_correction
  Model: BioMistral-7B
  Error: Empty or unreadable transcript
  Timestamp: 2025-08-13 07:51:40

PROCESSING STATISTICS:
  Total Files: 150
  Successful: 148
  Failed: 2
  Success Rate: 98.7%

ERROR BREAKDOWN:
  - Empty/unreadable files: 2
  - Model processing failures: 0
  - File save failures: 0
```

### Pipeline Summary Format
```
EMS Call ASR Pipeline Summary
=============================
Date: 2025-08-13 08:00:00
Input Directory: /path/to/audio
Output Directory: /path/to/results
Ground Truth File: /path/to/ground_truth.csv

Processing Configuration:
  - VAD Processing: Enabled
  - Long Audio Split: Enabled
  - Audio Filtering: Enabled
  - Ground Truth Preprocessing: Enabled

Processing Results:
  - Total audio files: 50
  - Successfully processed: 48
  - Failed processing: 2
  - Total processing time: 01:23:45

ASR Model Results:
  - Whisper Large-v3: 48/50 files (WER: 0.15)
  - Wav2Vec2: 47/50 files (WER: 0.22)
  - Canary-1B: 46/50 files (WER: 0.18)
  - Parakeet: 45/50 files (WER: 0.25)

Output Structure:
  /path/to/results/asr_transcripts/     # Raw ASR results
  /path/to/results/evaluation/          # Performance metrics
  /path/to/results/summary.txt          # This summary
```

## 🔍 File Validation and Quality Control

### Input Validation Scripts

#### Audio File Validation
```python
def validate_audio_file(file_path):
    """Validate audio file format and properties"""
    try:
        import librosa
        y, sr = librosa.load(file_path, sr=None)
        
        validation_result = {
            'file_path': file_path,
            'valid': True,
            'sample_rate': sr,
            'duration': len(y) / sr,
            'channels': 1 if len(y.shape) == 1 else y.shape[1],
            'issues': []
        }
        
        # Check sample rate
        if sr not in [8000, 16000, 22050, 44100, 48000]:
            validation_result['issues'].append(f'Unusual sample rate: {sr}Hz')
        
        # Check duration
        if len(y) / sr < 1.0:
            validation_result['issues'].append('Audio too short (<1 second)')
        elif len(y) / sr > 7200:  # 2 hours
            validation_result['issues'].append('Audio very long (>2 hours)')
        
        return validation_result
        
    except Exception as e:
        return {
            'file_path': file_path,
            'valid': False,
            'error': str(e)
        }
```

#### Ground Truth Validation
```python
def validate_ground_truth_csv(csv_path):
    """Validate ground truth CSV format"""
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
        
        validation_result = {
            'file_path': csv_path,
            'valid': True,
            'row_count': len(df),
            'issues': []
        }
        
        # Check required columns
        required_columns = ['Filename', 'transcript']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['valid'] = False
            validation_result['issues'].append(f'Missing columns: {missing_columns}')
        
        # Check for empty transcripts
        if 'transcript' in df.columns:
            empty_transcripts = df['transcript'].isna().sum()
            if empty_transcripts > 0:
                validation_result['issues'].append(f'Empty transcripts: {empty_transcripts}')
        
        # Check filename format
        if 'Filename' in df.columns:
            invalid_filenames = df[~df['Filename'].str.contains(r'\.(wav|mp3|flac|m4a)$', case=False)]
            if len(invalid_filenames) > 0:
                validation_result['issues'].append(f'Invalid filenames: {len(invalid_filenames)}')
        
        return validation_result
        
    except Exception as e:
        return {
            'file_path': csv_path,
            'valid': False,
            'error': str(e)
        }
```

### Output Quality Checks

#### Transcript Quality Validation
```python
def validate_transcript_quality(transcript_text):
    """Validate transcript content quality"""
    quality_metrics = {
        'length': len(transcript_text),
        'word_count': len(transcript_text.split()),
        'has_medical_terms': False,
        'readability_score': 0,
        'issues': []
    }
    
    # Check for medical terminology
    medical_terms = ['patient', 'medical', 'hospital', 'doctor', 'nurse', 'emergency', 'ambulance']
    if any(term in transcript_text.lower() for term in medical_terms):
        quality_metrics['has_medical_terms'] = True
    
    # Check for common issues
    if len(transcript_text.strip()) == 0:
        quality_metrics['issues'].append('Empty transcript')
    elif len(transcript_text.split()) < 5:
        quality_metrics['issues'].append('Very short transcript')
    
    # Check for repeated patterns
    words = transcript_text.lower().split()
    if len(set(words)) < len(words) * 0.5:  # More than 50% repeated words
        quality_metrics['issues'].append('High word repetition')
    
    return quality_metrics
```

## 🔗 Related Documentation

- [ASR Pipeline Guide](ASR_PIPELINE_GUIDE.md) - Processing workflow details
- [LLM Pipeline Guide](LLM_PIPELINE_GUIDE.md) - Enhancement workflow details
- [Error Handling Guide](ERROR_HANDLING_GUIDE.md) - Troubleshooting file issues
- [Command Reference](COMMAND_REFERENCE.md) - All available parameters

## 📞 Support

For file format issues:

1. **Validate input files**: Use provided validation scripts
2. **Check file permissions**: Ensure read/write access
3. **Verify file encoding**: Use UTF-8 for text files
4. **Test with sample data**: Use provided example files
5. **Review error logs**: Check detailed error messages

---

**Note**: This guide covers file formats for both pipeline stages. Always validate input files before processing to avoid issues during pipeline execution.