#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Analyzer for ASR Model Compatibility
==========================================

This script analyzes input audio files and generates detailed reports
about their compatibility with different ASR models.
"""

import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ASRModelRequirements:
    """ASR model requirements and limitations"""
    
    MODELS = {
        "whisper-large-v3": {
            "name": "Whisper (large-v3)",
            "description": "Most flexible model with minimal restrictions",
            "requirements": {
                "duration": {"min": 0.0, "max": float('inf'), "unit": "seconds"},
                "sample_rate": {"supported": "any", "preferred": "any"},
                "channels": {"supported": "any", "preferred": "any"},
                "volume": {"min": 0.0, "max": float('inf')},
                "format": ["wav", "mp3", "m4a", "flac", "ogg", "webm"]
            },
            "limitations": [],
            "recommendations": ["No preprocessing required for most files"]
        },
        
        "canary-1b": {
            "name": "Canary-1b (NeMo)",
            "description": "Strict limitations for optimal performance",
            "requirements": {
                "duration": {"min": 0.5, "max": 60.0, "unit": "seconds"},
                "sample_rate": {"supported": 16000, "preferred": 16000},
                "channels": {"supported": 1, "preferred": 1},
                "volume": {"min": 0.01, "max": float('inf')},
                "format": ["wav"]
            },
            "limitations": [
                "Duration must be between 0.5-60 seconds",
                "Sample rate must be 16kHz",
                "Audio must be mono (single channel)",
                "Minimum volume threshold of 0.01",
                "Only WAV format supported"
            ],
            "recommendations": [
                "Resample to 16kHz if different",
                "Convert to mono if stereo",
                "Normalize volume to minimum 0.01",
                "Split long audio into segments",
                "Pad short audio to minimum duration"
            ]
        },
        
        "parakeet-tdt-0.6b-v2": {
            "name": "Parakeet-tdt-0.6b-v2 (NeMo)",
            "description": "Moderate limitations with good flexibility",
            "requirements": {
                "duration": {"min": 1.0, "max": 300.0, "unit": "seconds"},
                "sample_rate": {"supported": 16000, "preferred": 16000},
                "channels": {"supported": 1, "preferred": 1},
                "volume": {"min": 0.0, "max": float('inf')},
                "format": ["wav"]
            },
            "limitations": [
                "Duration must be between 1.0-300 seconds",
                "Sample rate must be 16kHz",
                "Audio must be mono (single channel)",
                "Only WAV format supported"
            ],
            "recommendations": [
                "Resample to 16kHz if different",
                "Convert to mono if stereo",
                "Split very long audio into segments",
                "Pad short audio to minimum duration"
            ]
        },
        
        "wav2vec-xls-r": {
            "name": "Wav2Vec2-xls-r (Transformers)",
            "description": "Good flexibility with moderate requirements",
            "requirements": {
                "duration": {"min": 0.1, "max": float('inf'), "unit": "seconds"},
                "sample_rate": {"supported": 16000, "preferred": 16000},
                "channels": {"supported": "any", "preferred": 1},
                "volume": {"min": 0.01, "max": float('inf')},
                "format": ["wav", "mp3", "flac"]
            },
            "limitations": [
                "Duration must be at least 0.1 seconds",
                "Sample rate should be 16kHz",
                "Minimum volume threshold of 0.01",
                "Mono audio preferred"
            ],
            "recommendations": [
                "Resample to 16kHz if different",
                "Convert to mono if stereo",
                "Normalize volume to minimum 0.01"
            ]
        }
    }

class AudioAnalyzer:
    """Audio analyzer for ASR model compatibility"""
    
    def __init__(self):
        self.model_requirements = ASRModelRequirements.MODELS
    
    def analyze_audio_file(self, file_path: str) -> Dict:
        """Analyze a single audio file"""
        try:
            # Get audio info
            info = sf.info(file_path)
            
            # Read audio data for volume analysis
            audio, sample_rate = sf.read(file_path)
            
            # Calculate volume metrics
            if len(audio.shape) > 1:  # Multi-channel
                volume_rms = np.sqrt(np.mean(audio**2, axis=0))
                max_volume = np.max(np.abs(audio))
                avg_volume = np.mean(volume_rms)
            else:  # Mono
                volume_rms = np.sqrt(np.mean(audio**2))
                max_volume = np.max(np.abs(audio))
                avg_volume = volume_rms
            
            # Analyze compatibility with each model
            compatibility = {}
            for model_id, model_info in self.model_requirements.items():
                compatibility[model_id] = self._check_model_compatibility(
                    info, audio, model_id, model_info, file_path
                )
            
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size_bytes": os.path.getsize(file_path),
                "audio_info": {
                    "duration_seconds": info.duration,
                    "sample_rate_hz": info.samplerate,
                    "channels": info.channels,
                    "format": info.format,
                    "subtype": info.subtype
                },
                "volume_analysis": {
                    "max_volume": float(max_volume),
                    "avg_volume": float(avg_volume),
                    "volume_rms": float(volume_rms) if not isinstance(volume_rms, np.ndarray) else volume_rms.tolist()
                },
                "model_compatibility": compatibility
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "error": str(e),
                "model_compatibility": {}
            }
    
    def _check_model_compatibility(self, info, audio, model_id: str, model_info: Dict, file_path: str) -> Dict:
        """Check compatibility with a specific model"""
        requirements = model_info["requirements"]
        issues = []
        recommendations = []
        
        # Check duration
        duration = info.duration
        min_duration = requirements["duration"]["min"]
        max_duration = requirements["duration"]["max"]
        
        if duration < min_duration:
            issues.append(f"Duration too short: {duration:.2f}s (minimum: {min_duration}s)")
            recommendations.append(f"Pad audio to minimum {min_duration}s")
        elif duration > max_duration:
            issues.append(f"Duration too long: {duration:.2f}s (maximum: {max_duration}s)")
            recommendations.append(f"Split audio into segments of maximum {max_duration}s")
        
        # Check sample rate
        sample_rate = info.samplerate
        required_sr = requirements["sample_rate"]["supported"]
        if required_sr != "any" and sample_rate != required_sr:
            issues.append(f"Sample rate mismatch: {sample_rate}Hz (required: {required_sr}Hz)")
            recommendations.append(f"Resample to {required_sr}Hz")
        
        # Check channels
        channels = info.channels
        required_channels = requirements["channels"]["supported"]
        if required_channels != "any" and channels != required_channels:
            issues.append(f"Channel mismatch: {channels} channels (required: {required_channels})")
            recommendations.append(f"Convert to {required_channels} channel(s)")
        
        # Check volume
        if len(audio.shape) > 1:
            max_vol = np.max(np.abs(audio))
        else:
            max_vol = np.max(np.abs(audio))
        
        min_volume = requirements["volume"]["min"]
        if max_vol < min_volume:
            issues.append(f"Volume too low: {max_vol:.4f} (minimum: {min_volume})")
            recommendations.append(f"Normalize volume to minimum {min_volume}")
        
        # Check format
        file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        supported_formats = requirements["format"]
        if file_ext not in supported_formats:
            issues.append(f"Format not supported: {file_ext} (supported: {', '.join(supported_formats)})")
            recommendations.append(f"Convert to {supported_formats[0]} format")
        
        # Determine compatibility status
        is_compatible = len(issues) == 0
        
        return {
            "model_name": model_info["name"],
            "description": model_info["description"],
            "is_compatible": is_compatible,
            "issues": issues,
            "recommendations": recommendations + model_info["recommendations"],
            "limitations": model_info["limitations"]
        }
    
    def analyze_directory(self, input_dir: str) -> Dict:
        """Analyze all audio files in a directory"""
        input_path = Path(input_dir)
        audio_files = []
        
        # Find all audio files
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg', '*.webm']:
            audio_files.extend(input_path.glob(ext))
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return {"error": "No audio files found"}
        
        # Analyze each file
        results = []
        for file_path in sorted(audio_files):
            result = self.analyze_audio_file(str(file_path))
            results.append(result)
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "input_directory": input_dir,
            "total_files": len(results),
            "files": results
        }
    
    def generate_report(self, analysis_result: Dict, output_file: str = None) -> str:
        """Generate a detailed English report"""
        
        if "error" in analysis_result:
            return f"Error: {analysis_result['error']}"
        
        report = []
        report.append("=" * 80)
        report.append("AUDIO ANALYSIS REPORT FOR ASR MODEL COMPATIBILITY")
        report.append("=" * 80)
        report.append(f"Analysis Date: {analysis_result['analysis_date']}")
        report.append(f"Input Directory: {analysis_result['input_directory']}")
        report.append(f"Total Files Analyzed: {analysis_result['total_files']}")
        report.append("")
        
        # Summary statistics
        total_files = len(analysis_result['files'])
        compatible_files = {model_id: 0 for model_id in self.model_requirements.keys()}
        
        for file_result in analysis_result['files']:
            if 'error' in file_result:
                continue
            for model_id, compatibility in file_result['model_compatibility'].items():
                if compatibility['is_compatible']:
                    compatible_files[model_id] += 1
        
        report.append("COMPATIBILITY SUMMARY:")
        report.append("-" * 40)
        for model_id, count in compatible_files.items():
            model_name = self.model_requirements[model_id]['name']
            percentage = (count / total_files) * 100
            report.append(f"{model_name}: {count}/{total_files} files ({percentage:.1f}%)")
        report.append("")
        
        # Detailed analysis for each file
        for i, file_result in enumerate(analysis_result['files'], 1):
            if 'error' in file_result:
                report.append(f"FILE {i}: {file_result['file_name']}")
                report.append(f"ERROR: {file_result['error']}")
                report.append("")
                continue
            
            report.append(f"FILE {i}: {file_result['file_name']}")
            report.append("-" * 50)
            
            # Audio information
            info = file_result['audio_info']
            report.append(f"Duration: {info['duration_seconds']:.2f} seconds")
            report.append(f"Sample Rate: {info['sample_rate_hz']} Hz")
            report.append(f"Channels: {info['channels']}")
            report.append(f"Format: {info['format']}")
            report.append(f"File Size: {file_result['file_size_bytes']:,} bytes")
            
            # Volume information
            vol = file_result['volume_analysis']
            report.append(f"Max Volume: {vol['max_volume']:.4f}")
            report.append(f"Average Volume: {vol['avg_volume']:.4f}")
            report.append("")
            
            # Model compatibility
            report.append("MODEL COMPATIBILITY:")
            for model_id, compatibility in file_result['model_compatibility'].items():
                status = "? COMPATIBLE" if compatibility['is_compatible'] else "? INCOMPATIBLE"
                report.append(f"  {compatibility['model_name']}: {status}")
                
                if not compatibility['is_compatible']:
                    report.append(f"    Issues:")
                    for issue in compatibility['issues']:
                        report.append(f"      - {issue}")
                    report.append(f"    Recommendations:")
                    for rec in compatibility['recommendations']:
                        report.append(f"      - {rec}")
                report.append("")
            
            report.append("")
        
        # Overall recommendations
        report.append("OVERALL RECOMMENDATIONS:")
        report.append("-" * 40)
        
        # Collect common issues
        all_issues = []
        for file_result in analysis_result['files']:
            if 'error' in file_result:
                continue
            for compatibility in file_result['model_compatibility'].values():
                all_issues.extend(compatibility['issues'])
        
        if all_issues:
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            report.append("Common issues found:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  - {issue} ({count} files)")
            report.append("")
        
        # Model-specific recommendations
        report.append("MODEL-SPECIFIC RECOMMENDATIONS:")
        report.append("-" * 40)
        
        for model_id, model_info in self.model_requirements.items():
            report.append(f"{model_info['name']}:")
            report.append(f"  Description: {model_info['description']}")
            if model_info['limitations']:
                report.append("  Limitations:")
                for limitation in model_info['limitations']:
                    report.append(f"    - {limitation}")
            if model_info['recommendations']:
                report.append("  General Recommendations:")
                for rec in model_info['recommendations']:
                    report.append(f"    - {rec}")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_file}")
        
        return report_text

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Analyze audio files for ASR model compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', type=str,
                       help='Input audio file or directory')
    parser.add_argument('--output', '-o', type=str,
                       help='Output report file (optional)')
    parser.add_argument('--json', action='store_true',
                       help='Output JSON format instead of text report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize analyzer
    analyzer = AudioAnalyzer()
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Analyze single file
        result = analyzer.analyze_audio_file(args.input)
        analysis_result = {
            "analysis_date": datetime.now().isoformat(),
            "input_file": args.input,
            "total_files": 1,
            "files": [result]
        }
    elif os.path.isdir(args.input):
        # Analyze directory
        analysis_result = analyzer.analyze_directory(args.input)
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    # Generate report
    if args.json:
        # Output JSON
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON report saved to: {args.output}")
        else:
            print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
    else:
        # Output text report
        report = analyzer.generate_report(analysis_result, args.output)
        if not args.output:
            print(report)
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 