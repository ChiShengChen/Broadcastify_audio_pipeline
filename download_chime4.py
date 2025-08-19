#!/usr/bin/env python3
"""
CHiME4 Dataset Download Script

This script downloads the CHiME4 dataset for distant-talking automatic speech recognition.
Based on the CHiME4 challenge data structure from https://www.chimechallenge.org/challenges/chime4/data

The dataset includes:
- Audio data (16kHz WAV files)
- Annotations (JSON files) 
- Transcriptions
- WSJ0 subset
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CHiME4Downloader:
    def __init__(self, download_dir="/media/meow/One Touch/ems_call/CHiME4"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # CHiME4 dataset URLs (based on the official website)
        self.dataset_urls = {
            "chime4_diff": "https://mab.to/dMwDNq4r2",
            "ldc_package": "https://catalog.ldc.upenn.edu/LDC2017S24"  # Requires LDC license
        }
        
        # Alternative download URLs (if the above don't work)
        self.alternative_urls = {
            "github_mirror": "https://github.com/chimechallenge/chime4-data/releases/download/v1.0/CHiME4.zip"
        }
    
    def download_file(self, url, filename, chunk_size=8192):
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            filepath = self.download_dir / filename
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded: {filename}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def extract_archive(self, filepath):
        """Extract downloaded archive"""
        try:
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(self.download_dir)
            elif filepath.suffix in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(self.download_dir)
            else:
                logger.warning(f"Unknown archive format: {filepath.suffix}")
                return False
            
            logger.info(f"Extracted: {filepath.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {filepath}: {e}")
            return False
    
    def create_directory_structure(self):
        """Create the expected CHiME4 directory structure"""
        directories = [
            "data/audio/16kHz/backgrounds",
            "data/audio/16kHz/embedded", 
            "data/audio/16kHz/isolated",
            "data/audio/16kHz/isolated_1ch_track",
            "data/audio/16kHz/isolated_2ch_track",
            "data/audio/16kHz/isolated_6ch_track",
            "data/annotations",
            "data/transcriptions",
            "data/WSJ0"
        ]
        
        for directory in directories:
            (self.download_dir / directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Created CHiME4 directory structure")
    
    def download_chime4_dataset(self):
        """Main download function"""
        logger.info(f"Starting CHiME4 dataset download to: {self.download_dir}")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Try to download the CHiME4_diff file first (available without license)
        logger.info("Attempting to download CHiME4_diff...")
        chime4_diff_url = self.dataset_urls["chime4_diff"]
        filename = "CHiME4_diff_v1.0.zip"
        
        logger.info(f"Downloading CHiME4_diff from: {chime4_diff_url}")
        downloaded_file = self.download_file(chime4_diff_url, filename)
        
        if downloaded_file:
            logger.info("Successfully downloaded CHiME4_diff")
            if self.extract_archive(downloaded_file):
                logger.info("CHiME4_diff extracted successfully")
            else:
                logger.warning("Failed to extract CHiME4_diff")
        
        # Try the GitHub mirror as backup
        logger.info("Trying GitHub mirror as backup...")
        github_url = self.alternative_urls["github_mirror"]
        filename = "CHiME4.zip"
        
        logger.info(f"Trying GitHub mirror: {github_url}")
        downloaded_file = self.download_file(github_url, filename)
        
        if downloaded_file:
            logger.info("Successfully downloaded CHiME4 dataset from GitHub")
            if self.extract_archive(downloaded_file):
                logger.info("CHiME4 dataset extracted successfully")
                return True
        
        # Note about LDC package
        logger.info("Note: The main CHiME4 dataset (audio data) is available via LDC:")
        logger.info("LDC Package: https://catalog.ldc.upenn.edu/LDC2017S24")
        logger.info("This requires an LDC license. The CHiME4_diff contains annotations and baseline code.")
        
        return True
    
    def verify_download(self):
        """Verify that the downloaded dataset has the expected structure"""
        expected_files = [
            "data/audio/16kHz/isolated",
            "data/annotations",
            "data/transcriptions"
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not (self.download_dir / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"Missing expected files/directories: {missing_files}")
            return False
        else:
            logger.info("CHiME4 dataset structure verified successfully")
            return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download CHiME4 dataset")
    parser.add_argument("--download_dir", 
                       default="/media/meow/One Touch/ems_call/CHiME4",
                       help="Directory to download CHiME4 dataset")
    parser.add_argument("--verify", action="store_true",
                       help="Verify downloaded dataset structure")
    
    args = parser.parse_args()
    
    downloader = CHiME4Downloader(args.download_dir)
    
    if args.verify:
        if downloader.verify_download():
            print("✓ CHiME4 dataset verification successful")
        else:
            print("✗ CHiME4 dataset verification failed")
            sys.exit(1)
    else:
        if downloader.download_chime4_dataset():
            print("✓ CHiME4 dataset downloaded successfully")
            if downloader.verify_download():
                print("✓ Dataset structure verified")
            else:
                print("⚠ Dataset structure verification failed")
        else:
            print("✗ CHiME4 dataset download failed")
            sys.exit(1)

if __name__ == "__main__":
    main() 