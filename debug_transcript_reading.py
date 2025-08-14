#!/usr/bin/env python3
"""
Debug Transcript Reading Script

This script replicates the transcript discovery and reading logic from run_llm_pipeline.sh
to help debug transcript reading issues.

Based on the approach used in run_llm_pipeline.sh:
1. Look for transcripts in multiple possible locations
2. Filter for Whisper results (large-v3) if needed
3. Validate file contents and readability
4. Provide detailed diagnostics
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptDebugger:
    """Debug transcript reading issues using the same logic as run_llm_pipeline.sh"""
    
    def __init__(self, asr_results_dir: str, whisper_filter: bool = True):
        self.asr_results_dir = Path(asr_results_dir)
        self.whisper_filter = whisper_filter
        self.transcript_dirs = []
        self.transcript_files = []
        
    def step1_find_transcript_directories(self) -> List[Path]:
        """
        Step 1: Find ASR Transcripts - replicate the exact logic from run_llm_pipeline.sh
        """
        logger.info("=== Step 1: Locating ASR Transcripts ===")
        
        # Look for transcripts in various possible locations (same as bash script)
        possible_dirs = [
            self.asr_results_dir / "asr_transcripts",
            self.asr_results_dir / "merged_transcripts", 
            self.asr_results_dir / "merged_segmented_transcripts"
        ]
        
        transcript_dirs = []
        for dir_path in possible_dirs:
            if dir_path.exists() and dir_path.is_dir():
                transcript_dirs.append(dir_path)
                logger.info(f"Found transcript directory: {dir_path}")
        
        # If no specific transcript directory found, check the root
        if not transcript_dirs:
            logger.info("No specific transcript directories found, checking root directory...")
            # Check if there are .txt files in the root directory
            txt_files = list(self.asr_results_dir.glob("*.txt"))
            if txt_files:
                transcript_dirs.append(self.asr_results_dir)
                logger.info(f"Found .txt files in root directory: {self.asr_results_dir}")
            
        if not transcript_dirs:
            logger.error(f"No transcript directories found in {self.asr_results_dir}")
            logger.error("Expected locations:")
            for expected_dir in possible_dirs:
                logger.error(f"  - {expected_dir}")
            logger.error(f"  - {self.asr_results_dir}/*.txt (root directory)")
            return []
        
        logger.info("Found transcript directories:")
        for dir_path in transcript_dirs:
            logger.info(f"  - {dir_path}")
        
        self.transcript_dirs = transcript_dirs
        return transcript_dirs
    
    def count_transcript_files(self) -> int:
        """Count total transcript files - replicate bash script logic"""
        total_transcripts = 0
        for dir_path in self.transcript_dirs:
            txt_files = list(dir_path.glob("*.txt"))
            count = len(txt_files)
            total_transcripts += count
            logger.info(f"Directory {dir_path}: {count} .txt files")
        
        logger.info(f"Total transcript files found: {total_transcripts}")
        return total_transcripts
    
    def step1_5_filter_whisper_files(self) -> List[Path]:
        """
        Step 1.5: Whisper Filter - replicate the filtering logic from run_llm_pipeline.sh
        """
        if not self.whisper_filter:
            logger.info("=== Skipping Whisper Filter ===")
            return self._collect_all_transcript_files()
        
        logger.info("=== Step 1.5: Filtering Whisper Results ===")
        logger.info("Filtering Whisper (large-v3) results from transcript directories...")
        
        whisper_files = []
        for input_dir in self.transcript_dirs:
            if not input_dir.exists():
                logger.warning(f"Input directory does not exist: {input_dir}")
                continue
            
            # Find all large-v3 files (Whisper results) - same logic as bash script
            for file_path in input_dir.rglob("*.txt"):
                if "large-v3_" in file_path.name:
                    whisper_files.append(file_path)
                    logger.debug(f"Found Whisper file: {file_path.name}")
        
        logger.info(f"Found {len(whisper_files)} Whisper (large-v3) files")
        
        # List all found Whisper files
        if whisper_files:
            logger.info("Whisper files found:")
            for file_path in whisper_files:
                logger.info(f"  - {file_path}")
        else:
            logger.warning("No Whisper (large-v3) files found!")
            logger.warning("Available files in transcript directories:")
            for dir_path in self.transcript_dirs:
                txt_files = list(dir_path.glob("*.txt"))
                for txt_file in txt_files[:10]:  # Show first 10 files
                    logger.warning(f"  - {txt_file.name}")
                if len(txt_files) > 10:
                    logger.warning(f"  ... and {len(txt_files) - 10} more files")
        
        self.transcript_files = whisper_files
        return whisper_files
    
    def _collect_all_transcript_files(self) -> List[Path]:
        """Collect all transcript files without filtering"""
        all_files = []
        for dir_path in self.transcript_dirs:
            txt_files = list(dir_path.rglob("*.txt"))
            all_files.extend(txt_files)
        
        logger.info(f"Collected {len(all_files)} total transcript files")
        self.transcript_files = all_files
        return all_files
    
    def validate_transcript_files(self) -> Dict[str, any]:
        """Validate transcript files for readability and content"""
        logger.info("=== Validating Transcript Files ===")
        
        validation_results = {
            'total_files': len(self.transcript_files),
            'readable_files': 0,
            'empty_files': 0,
            'unreadable_files': 0,
            'file_details': [],
            'errors': []
        }
        
        for file_path in self.transcript_files:
            file_info = {
                'path': str(file_path),
                'name': file_path.name,
                'size': 0,
                'readable': False,
                'empty': False,
                'content_preview': '',
                'error': None
            }
            
            try:
                # Check file size
                file_info['size'] = file_path.stat().st_size
                
                # Try to read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if not content:
                    file_info['empty'] = True
                    validation_results['empty_files'] += 1
                    logger.warning(f"Empty file: {file_path.name}")
                else:
                    file_info['readable'] = True
                    file_info['content_preview'] = content[:100] + ('...' if len(content) > 100 else '')
                    validation_results['readable_files'] += 1
                    logger.debug(f"Readable file: {file_path.name} ({len(content)} chars)")
                    
            except Exception as e:
                file_info['error'] = str(e)
                validation_results['unreadable_files'] += 1
                validation_results['errors'].append(f"{file_path.name}: {e}")
                logger.error(f"Cannot read file {file_path.name}: {e}")
            
            validation_results['file_details'].append(file_info)
        
        # Summary
        logger.info("=== Validation Summary ===")
        logger.info(f"Total files: {validation_results['total_files']}")
        logger.info(f"Readable files: {validation_results['readable_files']}")
        logger.info(f"Empty files: {validation_results['empty_files']}")
        logger.info(f"Unreadable files: {validation_results['unreadable_files']}")
        
        if validation_results['errors']:
            logger.error("Errors encountered:")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
        
        return validation_results
    
    def analyze_file_patterns(self) -> Dict[str, any]:
        """Analyze file naming patterns to understand the structure"""
        logger.info("=== Analyzing File Patterns ===")
        
        patterns = {
            'whisper_large_v3': 0,
            'other_models': {},
            'file_extensions': {},
            'naming_patterns': []
        }
        
        for file_path in self.transcript_files:
            name = file_path.name
            
            # Count Whisper large-v3 files
            if "large-v3_" in name:
                patterns['whisper_large_v3'] += 1
            
            # Track other model patterns
            if "_" in name:
                parts = name.split("_")
                if len(parts) > 0:
                    potential_model = parts[0]
                    if potential_model not in patterns['other_models']:
                        patterns['other_models'][potential_model] = 0
                    patterns['other_models'][potential_model] += 1
            
            # Track extensions
            ext = file_path.suffix
            if ext not in patterns['file_extensions']:
                patterns['file_extensions'][ext] = 0
            patterns['file_extensions'][ext] += 1
            
            # Store naming pattern examples
            if len(patterns['naming_patterns']) < 10:
                patterns['naming_patterns'].append(name)
        
        logger.info(f"Whisper large-v3 files: {patterns['whisper_large_v3']}")
        logger.info("Other model patterns:")
        for model, count in patterns['other_models'].items():
            logger.info(f"  - {model}: {count} files")
        
        logger.info("File extensions:")
        for ext, count in patterns['file_extensions'].items():
            logger.info(f"  - {ext}: {count} files")
        
        logger.info("Example file names:")
        for name in patterns['naming_patterns']:
            logger.info(f"  - {name}")
        
        return patterns
    
    def debug_transcript_reading(self) -> Dict[str, any]:
        """Complete debugging workflow"""
        logger.info("Starting transcript reading debug process...")
        
        results = {
            'asr_results_dir': str(self.asr_results_dir),
            'whisper_filter_enabled': self.whisper_filter,
            'transcript_dirs': [],
            'total_files': 0,
            'validation': {},
            'patterns': {},
            'success': False
        }
        
        # Step 1: Find transcript directories
        transcript_dirs = self.step1_find_transcript_directories()
        if not transcript_dirs:
            results['error'] = "No transcript directories found"
            return results
        
        results['transcript_dirs'] = [str(d) for d in transcript_dirs]
        
        # Count files
        total_files = self.count_transcript_files()
        results['total_files'] = total_files
        
        if total_files == 0:
            results['error'] = "No transcript files found in directories"
            return results
        
        # Step 1.5: Filter Whisper files (if enabled)
        filtered_files = self.step1_5_filter_whisper_files()
        if not filtered_files:
            results['error'] = "No files found after filtering"
            return results
        
        # Validate files
        validation = self.validate_transcript_files()
        results['validation'] = validation
        
        # Analyze patterns
        patterns = self.analyze_file_patterns()
        results['patterns'] = patterns
        
        # Determine success
        if validation['readable_files'] > 0:
            results['success'] = True
            logger.info("✅ Transcript reading debug completed successfully!")
        else:
            results['error'] = "No readable transcript files found"
            logger.error("❌ No readable transcript files found")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Debug transcript reading issues")
    parser.add_argument("--asr_results_dir", required=True,
                       help="Directory containing ASR results")
    parser.add_argument("--enable_whisper_filter", action="store_true", default=True,
                       help="Enable filtering for Whisper results only (default: True)")
    parser.add_argument("--disable_whisper_filter", action="store_true",
                       help="Disable Whisper filtering")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--output_file", 
                       help="Save debug results to JSON file")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle whisper filter flag
    whisper_filter = args.enable_whisper_filter and not args.disable_whisper_filter
    
    # Validate input directory
    if not os.path.exists(args.asr_results_dir):
        logger.error(f"ASR results directory does not exist: {args.asr_results_dir}")
        sys.exit(1)
    
    # Run debug
    debugger = TranscriptDebugger(args.asr_results_dir, whisper_filter)
    results = debugger.debug_transcript_reading()
    
    # Save results if requested
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Debug results saved to: {args.output_file}")
    
    # Exit with appropriate code
    if results['success']:
        logger.info("Debug completed successfully")
        sys.exit(0)
    else:
        logger.error(f"Debug failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()