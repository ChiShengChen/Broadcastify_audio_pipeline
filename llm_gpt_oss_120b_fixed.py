#!/usr/bin/env python3
"""
Fixed version of gpt-oss-120b script with better error handling
Addresses the 'NoneType' object has no attribute 'to_dict' error
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_transformers_version():
    """Check if transformers version is compatible with gpt-oss-120b"""
    try:
        import transformers
        version = transformers.__version__
        logger.info(f"Transformers version: {version}")
        
        # Check for known compatible versions
        major, minor, patch = map(int, version.split('.')[:3])
        if major < 4 or (major == 4 and minor < 30):
            logger.warning(f"Transformers version {version} may have compatibility issues with gpt-oss-120b")
            logger.warning("Recommended: transformers >= 4.30.0")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking transformers version: {e}")
        return False

def safe_load_model(model_name: str, device: str = "auto", load_in_8bit: bool = False, load_in_4bit: bool = False):
    """Safely load the gpt-oss-120b model with error handling"""
    try:
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Model path: {model_name}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            device_map = "auto"
        else:
            logger.warning("CUDA not available, using CPU")
            device_map = "cpu"
        
        # Special handling for gpt-oss-120b
        logger.info("Special handling for gpt-oss-120b model")
        
        # Configure quantization
        quantization_config = None
        if load_in_8bit:
            logger.info("Loading in 8-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            logger.info("Loading in 4-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        else:
            logger.info("Skipping quantization for gpt-oss-120b")
        
        # Load tokenizer first
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with error handling
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        logger.info("Using auto device mapping for gpt-oss-120b")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        logger.error(f"Failed to create local model: {e}")
        
        # Try alternative loading method
        logger.info("Attempting alternative loading method...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                offload_folder="offload"  # Use disk offloading
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            return model, tokenizer
        except Exception as e2:
            logger.error(f"Alternative loading method also failed: {e2}")
            raise e2

def process_transcripts(input_dir: str, output_dir: str, prompt: str, temperature: float = 0.1, max_new_tokens: int = 128):
    """Process transcripts with the loaded model"""
    try:
        # Load model
        model, tokenizer = safe_load_model("openai/gpt-oss-120b")
        
        # Process files
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all transcript files
        transcript_files = list(input_path.rglob("*.txt"))
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        processed_count = 0
        for file_path in transcript_files:
            try:
                # Read transcript
                with open(file_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                
                if not transcript:
                    logger.warning(f"Empty transcript file: {file_path}")
                    continue
                
                # Prepare input
                full_prompt = f"{prompt}\n\n{transcript}"
                
                # Tokenize
                inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the generated part (remove the original prompt)
                response = generated_text[len(full_prompt):].strip()
                
                # Save result
                relative_path = file_path.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response)
                
                processed_count += 1
                logger.info(f"Processed: {relative_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {processed_count} files")
        return processed_count
        
    except Exception as e:
        logger.error(f"Error in process_transcripts: {e}")
        raise e

def main():
    if len(sys.argv) != 6:
        print("Usage: python3 llm_gpt_oss_120b_fixed.py <input_dir> <output_dir> <prompt> <temperature> <max_new_tokens>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    prompt = sys.argv[3]
    temperature = float(sys.argv[4])
    max_new_tokens = int(sys.argv[5])
    
    # Check transformers version
    if not check_transformers_version():
        logger.warning("Transformers version may cause compatibility issues")
    
    try:
        processed_count = process_transcripts(input_dir, output_dir, prompt, temperature, max_new_tokens)
        print(f"Successfully processed {processed_count} files")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 