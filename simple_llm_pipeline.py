#!/usr/bin/env python3
"""
Simple LLM Pipeline for Whisper Results

This script provides a simplified version of the LLM pipeline using smaller models
that can run on CPU without requiring large downloads.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import concurrent.futures
from tqdm import tqdm

# Import required libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError as e:
    print(f"Error: Required libraries not installed. Please install: {e}")
    print("Run: pip install torch transformers")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_llm_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleLLMModel:
    """Simple LLM model for inference using smaller models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading simple model: {self.model_name}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            logger.info("Creating inference pipeline...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device="cpu",
                torch_dtype=torch.float32
            )
            
            logger.info(f"Simple model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate response from the model"""
        for attempt in range(max_retries):
            try:
                # Prepare input
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                      max_length=512)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                # Decode output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the generated part (remove input prompt)
                response = generated_text[len(prompt):].strip()
                
                if response:
                    return response
                else:
                    logger.warning(f"Empty response from model on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} generation attempts failed")
                    return None
        
        return None

def load_transcript(file_path: Path) -> str:
    """Load transcript from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error loading transcript {file_path}: {e}")
        return ""

def save_corrected_transcript(content: str, output_path: Path):
    """Save corrected transcript to file"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error saving corrected transcript {output_path}: {e}")
        return False

def save_emergency_page(content: str, output_path: Path):
    """Save emergency page to file"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error saving emergency page {output_path}: {e}")
        return False

def correct_medical_terms(transcript: str, model: SimpleLLMModel) -> Optional[str]:
    """Correct medical terms in transcript using simple model"""
    try:
        # Create prompt with transcript
        prompt = f"Correct any medical terms in this transcript: {transcript}\nCorrected:"
        
        # Generate correction
        corrected = model.generate(prompt)
        if corrected:
            return corrected.strip()
        else:
            logger.warning("Model returned empty response")
            return None
    except Exception as e:
        logger.error(f"Error correcting medical terms: {e}")
        return None

def generate_emergency_page(transcript: str, model: SimpleLLMModel) -> Optional[str]:
    """Generate emergency page from transcript using simple model"""
    try:
        # Create prompt with transcript
        prompt = f"Generate an emergency page from this medical transcript: {transcript}\nEmergency Page:"
        
        # Generate emergency page
        emergency_page = model.generate(prompt)
        if emergency_page:
            return emergency_page.strip()
        else:
            logger.warning("Model returned empty response")
            return None
    except Exception as e:
        logger.error(f"Error generating emergency page: {e}")
        return None

def process_single_file_medical_correction(args: tuple) -> Dict[str, Any]:
    """Process a single transcript file for medical correction"""
    file_path, output_dir, model = args
    
    try:
        # Load original transcript
        original_transcript = load_transcript(file_path)
        if not original_transcript:
            return {
                'file': str(file_path),
                'success': False,
                'error': 'Empty or unreadable transcript'
            }
        
        # Correct medical terms
        corrected_transcript = correct_medical_terms(original_transcript, model)
        if corrected_transcript is None:
            return {
                'file': str(file_path),
                'success': False,
                'error': 'Model correction failed'
            }
        
        # Save corrected transcript
        output_path = output_dir / file_path.name
        if save_corrected_transcript(corrected_transcript, output_path):
            return {
                'file': str(file_path),
                'success': True,
                'original_length': len(original_transcript),
                'corrected_length': len(corrected_transcript)
            }
        else:
            return {
                'file': str(file_path),
                'success': False,
                'error': 'Failed to save corrected transcript'
            }
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {
            'file': str(file_path),
            'success': False,
            'error': str(e)
        }

def process_single_file_emergency_page(args: tuple) -> Dict[str, Any]:
    """Process a single transcript file for emergency page generation"""
    file_path, output_dir, model = args
    
    try:
        # Load transcript
        transcript = load_transcript(file_path)
        if not transcript:
            return {
                'file': str(file_path),
                'success': False,
                'error': 'Empty or unreadable transcript'
            }
        
        # Generate emergency page
        emergency_page = generate_emergency_page(transcript, model)
        if emergency_page is None:
            return {
                'file': str(file_path),
                'success': False,
                'error': 'Model page generation failed'
            }
        
        # Save emergency page
        output_path = output_dir / f"{file_path.stem}_emergency_page.txt"
        if save_emergency_page(emergency_page, output_path):
            return {
                'file': str(file_path),
                'success': True,
                'transcript_length': len(transcript),
                'page_length': len(emergency_page),
                'output_file': str(output_path)
            }
        else:
            return {
                'file': str(file_path),
                'success': False,
                'error': 'Failed to save emergency page'
            }
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {
            'file': str(file_path),
            'success': False,
            'error': str(e)
        }

def find_transcript_files(input_dirs: List[str]) -> List[Path]:
    """Find all transcript files in input directories"""
    transcript_files = []
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            continue
        
        # Find all .txt files
        for txt_file in input_path.rglob("*.txt"):
            transcript_files.append(txt_file)
    
    return transcript_files

def main():
    parser = argparse.ArgumentParser(description="Simple LLM Pipeline for Whisper Results")
    parser.add_argument("--mode", required=True, choices=["medical_correction", "emergency_page"],
                       help="Processing mode")
    parser.add_argument("--input_dirs", nargs="+", required=True,
                       help="Input directories containing transcript files")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--model", default="microsoft/DialoGPT-small",
                       help="Model to use (default: microsoft/DialoGPT-small)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Number of files to process in parallel")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create simple model
    try:
        model = SimpleLLMModel(args.model)
        logger.info(f"Created simple model: {args.model}")
    except Exception as e:
        logger.error(f"Failed to create simple model: {e}")
        sys.exit(1)
    
    # Find transcript files
    transcript_files = find_transcript_files(args.input_dirs)
    if not transcript_files:
        logger.error("No transcript files found in input directories")
        sys.exit(1)
    
    logger.info(f"Found {len(transcript_files)} transcript files to process")
    
    # Process files
    results = []
    successful = 0
    failed = 0
    
    # Choose processing function based on mode
    if args.mode == "medical_correction":
        process_func = process_single_file_medical_correction
        mode_name = "medical correction"
    else:
        process_func = process_single_file_emergency_page
        mode_name = "emergency page generation"
    
    # Prepare arguments for processing
    process_args = [
        (file_path, output_dir, model)
        for file_path in transcript_files
    ]
    
    # Process files
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_func, args): args[0]
            for args in process_args
        }
        
        # Process results as they complete
        with tqdm(total=len(transcript_files), desc=f"Processing {mode_name}") as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"Failed to process {result['file']}: {result.get('error', 'Unknown error')}")
                
                pbar.update(1)
    
    # Save processing summary
    summary = {
        'mode': args.mode,
        'model': args.model,
        'input_directories': args.input_dirs,
        'output_directory': str(output_dir),
        'total_files': len(transcript_files),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(transcript_files) if transcript_files else 0,
        'results': results
    }
    
    summary_file = output_dir / f"simple_{args.mode}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Log final results
    logger.info(f"Processing completed:")
    logger.info(f"  Total files: {len(transcript_files)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Success rate: {summary['success_rate']:.2%}")
    logger.info(f"  Summary saved to: {summary_file}")
    
    if failed > 0:
        logger.warning(f"{failed} files failed to process. Check the summary for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 