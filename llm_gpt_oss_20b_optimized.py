#!/usr/bin/env python3
"""
Optimized GPT-OSS-20B Model Handler with Advanced Memory Management
This script includes aggressive memory optimization for large models.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import sys
import logging
import gc
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedGPTOSS20BModel:
    def __init__(self, device="cuda", temperature=0.1, max_new_tokens=128, load_in_8bit=True):
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.tokenizer = None
        
    def aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup"""
        logger.info("Performing aggressive memory cleanup...")
        
        # Clear PyTorch cache multiple times
        if torch.cuda.is_available():
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(0.5)
            
            # Reset memory stats
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.reset_accumulated_memory_stats(i)
            
            torch.cuda.ipc_collect()
        
        # Aggressive garbage collection
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)
            
        logger.info("Memory cleanup completed")
    
    def load_model(self):
        """Load the model with optimized memory settings"""
        self.aggressive_memory_cleanup()
        
        logger.info("Loading gpt-oss-20b model: openai/gpt-oss-20b")
        logger.info("Cleared PyTorch CUDA cache")
        
        try:
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if self.load_in_8bit:
                logger.info("Using 8-bit quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
            
            # Load model with aggressive memory optimization
            logger.info("Loading model...")
            
            # Calculate memory allocation
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Use 80% of GPU memory for model, 20% for operations
                max_memory = {0: f"{int(total_memory * 0.8)}GB"}
                logger.info(f"Using max memory allocation: {max_memory}")
            else:
                max_memory = None
            
            # Model loading with optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                "openai/gpt-oss-20b",
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                max_memory=max_memory,
                torch_dtype=torch.float16 if not self.load_in_8bit else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                offload_folder="./offload",  # CPU offload directory
                offload_state_dict=True,
            )
            
            # Additional memory optimizations
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("gpt-oss-20b model loaded successfully!")
            
            # Final memory cleanup
            self.aggressive_memory_cleanup()
            
            # Report memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.aggressive_memory_cleanup()
            raise e
    
    def generate_text(self, prompt):
        """Generate text with memory-efficient settings"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Generate with memory-efficient settings
            with torch.no_grad():
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.nullcontext():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True if self.temperature > 0 else False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_return_sequences=1,
                        early_stopping=True,
                    )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Clean up tensors
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.aggressive_memory_cleanup()
            raise e
    
    def cleanup(self):
        """Clean up model and free memory"""
        logger.info("Cleaning up model...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.aggressive_memory_cleanup()
        logger.info("Model cleanup completed")

def process_transcripts(input_dir, output_dir, prompt, temperature=0.1, max_new_tokens=128):
    """Process transcripts with the optimized model"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all transcript files
    transcript_files = list(input_path.glob("*.txt"))
    
    if not transcript_files:
        logger.error("No transcript files found")
        return
    
    logger.info(f"Found {len(transcript_files)} transcript files to process")
    
    # Initialize model
    model = OptimizedGPTOSS20BModel(
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        load_in_8bit=True  # Always use 8-bit for memory efficiency
    )
    
    try:
        # Load model
        model.load_model()
        
        # Process each file
        for i, file_path in enumerate(transcript_files):
            try:
                logger.info(f"Processing file {i+1}/{len(transcript_files)}: {file_path.name}")
                
                # Read transcript
                with open(file_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                
                if not transcript:
                    logger.warning(f"Empty transcript file: {file_path.name}")
                    continue
                
                # Generate response
                full_prompt = f"{prompt}\n\n{transcript}"
                response = model.generate_text(full_prompt)
                
                # Save result
                output_file = output_path / file_path.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response)
                
                logger.info(f"Processed and saved: {output_file.name}")
                
                # Memory cleanup between files
                if i % 5 == 0:  # Every 5 files
                    model.aggressive_memory_cleanup()
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue
    
    finally:
        # Cleanup
        model.cleanup()

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 llm_gpt_oss_20b_optimized.py <input_dir> <output_dir> <prompt> <temperature> <max_new_tokens>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    prompt = sys.argv[3]
    temperature = float(sys.argv[4])
    max_new_tokens = int(sys.argv[5])
    
    # Set environment optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    logger.info(f"Using temperature: {temperature}, max_new_tokens: {max_new_tokens}")
    logger.info("Initializing optimized gpt-oss-20b model...")
    
    process_transcripts(input_dir, output_dir, prompt, temperature, max_new_tokens)

if __name__ == "__main__":
    main()
