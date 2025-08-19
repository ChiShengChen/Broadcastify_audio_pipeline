#!/usr/bin/env python3
"""
CHiME4 Manual Download Helper Script

This script provides instructions for manually downloading the CHiME4 dataset
and helps verify the dataset structure once downloaded.

Since the official download links are not working, this script provides
alternative approaches and verification tools.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CHiME4ManualDownloader:
    def __init__(self, download_dir="/media/meow/One Touch/ems_call/CHiME4"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected directory structure
        self.expected_structure = {
            "data/audio/16kHz/backgrounds": "Background noise recordings",
            "data/audio/16kHz/embedded": "Unsegmented noisy speech data",
            "data/audio/16kHz/isolated": "Segmented noisy speech data",
            "data/audio/16kHz/isolated_1ch_track": "1-channel track data",
            "data/audio/16kHz/isolated_2ch_track": "2-channel track data",
            "data/audio/16kHz/isolated_6ch_track": "6-channel track data",
            "data/annotations": "JSON annotation files",
            "data/transcriptions": "Transcription files",
            "data/WSJ0": "WSJ0 subset"
        }
        
        # Expected file patterns
        self.expected_files = {
            "data/annotations/dt05_real.json": "Development set real annotations",
            "data/annotations/dt05_simu.json": "Development set simulated annotations",
            "data/annotations/tr05_simu.json": "Training set simulated annotations",
            "data/annotations/mic_error.csv": "Microphone error data",
            "data/transcriptions/dt05_real.dot_all": "Development set real transcriptions",
            "data/transcriptions/dt05_simu.dot_all": "Development set simulated transcriptions"
        }
    
    def print_download_instructions(self):
        """Print comprehensive download instructions"""
        print("=" * 80)
        print("CHiME4 DATASET MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 80)
        print()
        
        print("1. OFFICIAL LDC PACKAGE (RECOMMENDED)")
        print("-" * 50)
        print("URL: https://catalog.ldc.upenn.edu/LDC2017S24")
        print("Requirements: LDC license (paid)")
        print("Content: Complete audio data, annotations, and baseline software")
        print()
        
        print("2. CONTACT CHiME ORGANIZERS")
        print("-" * 50)
        print("Email: chimechallenge@gmail.com")
        print("Request: Access to CHiME4 dataset")
        print("Mention: Whether you have a WSJ license")
        print()
        
        print("3. ACADEMIC INSTITUTION ACCESS")
        print("-" * 50)
        print("Many universities have access to LDC datasets.")
        print("Check with your institution's library or research computing center.")
        print()
        
        print("4. ALTERNATIVE SOURCES")
        print("-" * 50)
        print("- Check arXiv papers that use CHiME4")
        print("- Look for GitHub repositories with sample data")
        print("- Contact researchers who have used CHiME4")
        print()
        
        print("5. DATASET STRUCTURE")
        print("-" * 50)
        print("Expected directory structure:")
        for path, description in self.expected_structure.items():
            print(f"  {path}/ - {description}")
        print()
        
        print("6. VERIFICATION")
        print("-" * 50)
        print("After downloading, run this script with --verify to check the structure.")
        print()
        
        print("=" * 80)
    
    def verify_dataset_structure(self):
        """Verify that the downloaded dataset has the expected structure"""
        print("Verifying CHiME4 dataset structure...")
        print(f"Checking directory: {self.download_dir}")
        print()
        
        missing_dirs = []
        missing_files = []
        found_dirs = []
        found_files = []
        
        # Check directories
        for dir_path, description in self.expected_structure.items():
            full_path = self.download_dir / dir_path
            if full_path.exists():
                found_dirs.append((dir_path, description))
                print(f"✓ {dir_path}/ - {description}")
            else:
                missing_dirs.append((dir_path, description))
                print(f"✗ {dir_path}/ - {description} (MISSING)")
        
        print()
        
        # Check important files
        print("Checking important files:")
        for file_path, description in self.expected_files.items():
            full_path = self.download_dir / file_path
            if full_path.exists():
                found_files.append((file_path, description))
                print(f"✓ {file_path} - {description}")
            else:
                missing_files.append((file_path, description))
                print(f"✗ {file_path} - {description} (MISSING)")
        
        print()
        
        # Summary
        print("SUMMARY:")
        print(f"Found directories: {len(found_dirs)}/{len(self.expected_structure)}")
        print(f"Found files: {len(found_files)}/{len(self.expected_files)}")
        
        if missing_dirs or missing_files:
            print("\nMISSING COMPONENTS:")
            if missing_dirs:
                print("Missing directories:")
                for dir_path, description in missing_dirs:
                    print(f"  - {dir_path}/ - {description}")
            
            if missing_files:
                print("Missing files:")
                for file_path, description in missing_files:
                    print(f"  - {file_path} - {description}")
            
            print("\nThe dataset appears to be incomplete.")
            print("Please ensure you have downloaded the complete CHiME4 dataset.")
            return False
        else:
            print("\n✓ Dataset structure appears complete!")
            return True
    
    def analyze_audio_files(self):
        """Analyze audio files if present"""
        audio_dir = self.download_dir / "data/audio/16kHz/isolated"
        
        if not audio_dir.exists():
            print("Audio directory not found. Cannot analyze audio files.")
            return
        
        print("Analyzing audio files...")
        
        # Count files by environment
        environments = {}
        total_files = 0
        
        for subdir in audio_dir.iterdir():
            if subdir.is_dir():
                env_name = subdir.name
                file_count = len(list(subdir.glob("*.wav")))
                environments[env_name] = file_count
                total_files += file_count
        
        print(f"Total audio files found: {total_files}")
        print("Files by environment:")
        for env, count in sorted(environments.items()):
            print(f"  {env}: {count} files")
    
    def create_sample_structure(self):
        """Create a sample directory structure for reference"""
        print("Creating sample directory structure...")
        
        for dir_path in self.expected_structure.keys():
            full_path = self.download_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create a README file in each directory
            readme_path = full_path / "README.txt"
            if not readme_path.exists():
                with open(readme_path, 'w') as f:
                    f.write(f"Expected content: {self.expected_structure[dir_path]}\n")
                    f.write("This directory should contain the corresponding data files.\n")
        
        print(f"Sample structure created in: {self.download_dir}")
        print("This is for reference only - you still need to download the actual data.")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CHiME4 Manual Download Helper")
    parser.add_argument("--download_dir", 
                       default="/media/meow/One Touch/ems_call/CHiME4",
                       help="Directory for CHiME4 dataset")
    parser.add_argument("--instructions", action="store_true",
                       help="Print download instructions")
    parser.add_argument("--verify", action="store_true",
                       help="Verify downloaded dataset structure")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze audio files if present")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample directory structure")
    
    args = parser.parse_args()
    
    downloader = CHiME4ManualDownloader(args.download_dir)
    
    if args.instructions:
        downloader.print_download_instructions()
    
    if args.verify:
        downloader.verify_dataset_structure()
    
    if args.analyze:
        downloader.analyze_audio_files()
    
    if args.create_sample:
        downloader.create_sample_structure()
    
    # If no specific action is requested, show instructions
    if not any([args.instructions, args.verify, args.analyze, args.create_sample]):
        downloader.print_download_instructions()
        print("\nUse --help to see all available options.")

if __name__ == "__main__":
    main() 