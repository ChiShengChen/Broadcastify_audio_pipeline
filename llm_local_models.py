#!/usr/bin/env python3
"""
Local LLM Models for Medical Term Correction and Emergency Page Generation

This script loads Hugging Face models locally and provides inference capabilities
for medical term correction and emergency page generation.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import concurrent.futures
from tqdm import tqdm

# Import required libraries for local model inference
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers import BitsAndBytesConfig
    import accelerate
except ImportError as e:
    print(f"Error: Required libraries not installed. Please install: {e}")
    print("Run: pip install torch transformers accelerate bitsandbytes")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_local_models.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LocalLLMModel:
    """Local LLM model for inference"""
    
    def __init__(self, model_name: str, model_path: str = None, device: str = "auto", 
                 load_in_8bit: bool = False, load_in_4bit: bool = False):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Model configurations - optimized for BioMistral-7B
        self.model_configs = {
            "gpt-oss-20b": {
                "max_length": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": True
            },
            "BioMistral-7B": {
                "max_length": 3072,  # Optimized for BioMistral-7B's 4096 context
                "temperature": 0.3,  # Better for medical term generation
                "top_p": 0.9,
                "do_sample": True
            },
            "Meditron-7B": {
                "max_length": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": True
            },
            "Llama-3-8B-UltraMedica": {
                "max_length": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Model path: {self.model_path}")
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info("Using CUDA device")
                else:
                    self.device = "cpu"
                    logger.info("Using CPU device")
            
            # Configure quantization
            quantization_config = None
            if self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using 8-bit quantization")
            elif self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                logger.info("Using 4-bit quantization")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device if self.device == "cuda" else None,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                # If loading fails and we're not using quantization, try with 8-bit quantization
                if not self.load_in_8bit and not self.load_in_4bit and "accelerate" in str(e).lower():
                    logger.info("Model loading failed, trying with 8-bit quantization...")
                    self.load_in_8bit = True
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto",  # Use auto device mapping for quantized models
                        quantization_config=quantization_config,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    # Update device to match the model's actual device
                    if hasattr(self.model, 'device'):
                        self.device = str(self.model.device)
                    elif hasattr(self.model, 'hf_device_map'):
                        # For models with device mapping, use the first device
                        first_device = next(iter(self.model.hf_device_map.values()))
                        self.device = first_device if first_device != 'cpu' else 'cpu'
                else:
                    raise
            
            # Create pipeline
            logger.info("Creating inference pipeline...")
            # When using quantization (8-bit or 4-bit), don't specify device in pipeline
            if self.load_in_8bit or self.load_in_4bit:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device if self.device == "cpu" else 0,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            logger.info(f"Model {self.model_name} loaded successfully")
            logger.info(f"Model device: {getattr(self.model, 'device', 'unknown')}")
            logger.info(f"Current device setting: {self.device}")
            logger.info(f"Using quantization: 8-bit={self.load_in_8bit}, 4-bit={self.load_in_4bit}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate response from the model"""
        config = self.model_configs.get(self.model_name, {})
        
        for attempt in range(max_retries):
            try:
                # Prepare input with aggressive length limits for BioMistral-7B
                # BioMistral-7B has 4096 context length, but we need to leave room for generation
                max_input_length = 3072  # Leave 1024 tokens for generation
                
                # First, check if prompt is too long and truncate the transcript part
                if len(prompt) > 8000:  # Rough character limit
                    logger.warning(f"Prompt too long ({len(prompt)} chars), truncating transcript")
                    # Find the transcript part and truncate it
                    if "TRANSCRIPT:" in prompt:
                        parts = prompt.split("TRANSCRIPT:")
                        if len(parts) > 1:
                            instruction = parts[0]
                            transcript = parts[1]
                            # Truncate transcript to ~6000 characters
                            transcript = transcript[:6000] + "..." if len(transcript) > 6000 else transcript
                            prompt = instruction + "TRANSCRIPT:" + transcript
                
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                      max_length=max_input_length, padding=True)
                
                # Check input shape and log for debugging
                input_shape = inputs['input_ids'].shape
                logger.debug(f"Input shape: {input_shape}")
                if input_shape[1] > max_input_length:
                    logger.warning(f"Input too long ({input_shape[1]} tokens), truncating to {max_input_length}")
                    inputs['input_ids'] = inputs['input_ids'][:, :max_input_length]
                    inputs['attention_mask'] = inputs['attention_mask'][:, :max_input_length]
                
                # For quantized models, we need to move inputs to the same device as the model
                if self.device == "cuda":
                    if self.load_in_8bit or self.load_in_4bit:
                        # For quantized models, move to the same device as the model
                        if hasattr(self.model, 'device'):
                            target_device = self.model.device
                            logger.debug(f"Moving inputs to model device: {target_device}")
                            inputs = {k: v.to(target_device) for k, v in inputs.items()}
                        else:
                            # Fallback to cuda:0 for quantized models
                            logger.debug("Moving inputs to cuda:0 (fallback)")
                            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
                    else:
                        # For non-quantized models, move to the specified device
                        logger.debug(f"Moving inputs to device: {self.device}")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate with BioMistral-7B optimized parameters
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,  # Shorter output for medical corrections
                        temperature=0.3,  # Slightly higher for better medical term generation
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        num_beams=1,  # Use greedy decoding for stability
                        length_penalty=1.0
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
                error_msg = str(e)
                if "shapes cannot be multiplied" in error_msg:
                    logger.warning(f"Generation attempt {attempt + 1} failed: Matrix shape mismatch - input may be too long")
                    logger.debug(f"Input length: {len(prompt)} characters")
                    logger.debug(f"Error details: {error_msg}")
                    # Try with even shorter input on next attempt
                    if attempt < max_retries - 1:
                        # Truncate prompt more aggressively
                        if len(prompt) > 4000:
                            if "TRANSCRIPT:" in prompt:
                                parts = prompt.split("TRANSCRIPT:")
                                if len(parts) > 1:
                                    instruction = parts[0]
                                    transcript = parts[1]
                                    transcript = transcript[:3000] + "..." if len(transcript) > 3000 else transcript
                                    prompt = instruction + "TRANSCRIPT:" + transcript
                elif "out of memory" in error_msg.lower():
                    logger.warning(f"Generation attempt {attempt + 1} failed: Out of memory")
                    # Try with smaller batch or shorter input
                else:
                    logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} generation attempts failed")
                    return None
        
        return None

def create_local_model(model_name: str, model_path: str = None, device: str = "auto",
                      load_in_8bit: bool = False, load_in_4bit: bool = False) -> LocalLLMModel:
    """Create a local model instance"""
    
    # Model path mappings (if not provided)
    model_paths = {
        "gpt-oss-20b": "microsoft/DialoGPT-medium",  # Placeholder, replace with actual model
        "BioMistral-7B": "BioMistral/BioMistral-7B",
        "Meditron-7B": "epfl-llm/meditron-7b",
        "Llama-3-8B-UltraMedica": "meta-llama/Llama-3-8B"  # Placeholder, replace with actual model
    }
    
    if model_path is None:
        model_path = model_paths.get(model_name, model_name)
    
    return LocalLLMModel(model_name, model_path, device, load_in_8bit, load_in_4bit)

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

def correct_medical_terms(transcript: str, model: LocalLLMModel, prompt_template: str) -> Optional[str]:
    """Correct medical terms in transcript using local model"""
    try:
        # Create prompt with transcript
        prompt = f"{prompt_template}\n\nTranscript: {transcript}\n\nCorrected transcript:"
        
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

def generate_emergency_page(transcript: str, model: LocalLLMModel, prompt_template: str) -> Optional[str]:
    """Generate emergency page from transcript using local model"""
    try:
        # Create prompt with transcript
        prompt = f"{prompt_template}\n\nMedical Transcript: {transcript}\n\nEmergency Page:"
        
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
    file_path, output_dir, model, prompt_template = args
    
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
        corrected_transcript = correct_medical_terms(original_transcript, model, prompt_template)
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
    file_path, output_dir, model, prompt_template = args
    
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
        emergency_page = generate_emergency_page(transcript, model, prompt_template)
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
    parser = argparse.ArgumentParser(description="Local LLM Models for Medical Processing")
    parser.add_argument("--mode", required=True, choices=["medical_correction", "emergency_page"],
                       help="Processing mode")
    parser.add_argument("--input_dirs", nargs="+", required=True,
                       help="Input directories containing transcript files")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--model", required=True,
                       choices=["gpt-oss-20b", "BioMistral-7B", "Meditron-7B", "Llama-3-8B-UltraMedica"],
                       help="LLM model to use")
    parser.add_argument("--model_path", default=None,
                       help="Custom model path (optional)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to use for inference")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Number of files to process in parallel (note: limited by GPU memory)")
    parser.add_argument("--prompt", required=True,
                       help="Prompt template for processing")
    parser.add_argument("--error_log", default=None,
                       help="Path to error analysis log file (optional)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create local model
    try:
        model = create_local_model(
            args.model, 
            args.model_path, 
            args.device, 
            args.load_in_8bit, 
            args.load_in_4bit
        )
        logger.info(f"Created local model: {args.model}")
    except Exception as e:
        logger.error(f"Failed to create local model: {e}")
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
        (file_path, output_dir, model, args.prompt)
        for file_path in transcript_files
    ]
    
    # Process files (note: batch_size is limited for local models due to memory constraints)
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
                    error_msg = result.get('error', 'Unknown error')
                    logger.warning(f"Failed to process {result['file']}: {error_msg}")
                    
                    # Write to error log if specified
                    if args.error_log:
                        try:
                            with open(args.error_log, 'a', encoding='utf-8') as f:
                                f.write(f"FAILED FILE: {result['file']}\n")
                                f.write(f"  Processing Mode: {args.mode}\n")
                                f.write(f"  Model: {args.model}\n")
                                f.write(f"  Error: {error_msg}\n")
                                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write("\n")
                        except Exception as e:
                            logger.warning(f"Failed to write to error log: {e}")
                
                pbar.update(1)
    
    # Save processing summary
    summary = {
        'mode': args.mode,
        'model': args.model,
        'model_path': args.model_path or args.model,
        'device': args.device,
        'input_directories': args.input_dirs,
        'output_directory': str(output_dir),
        'total_files': len(transcript_files),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(transcript_files) if transcript_files else 0,
        'results': results
    }
    
    summary_file = output_dir / f"local_{args.mode}_summary.json"
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