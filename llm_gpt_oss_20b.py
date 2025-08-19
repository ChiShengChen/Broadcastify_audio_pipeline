#!/usr/bin/env python3
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import concurrent.futures

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPTOSS20BModel:
    """Class specifically for handling gpt-oss-20b model"""
    
    def __init__(self, device="cuda", temperature=0.1, max_new_tokens=128):
        self.device = device
        self.model_name = "openai/gpt-oss-20b"
        self.tokenizer = None
        self.model = None
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.load_model()
    
    def load_model(self):
        """Load the model"""
        try:
            logger.info(f"Loading gpt-oss-20b model: {self.model_name}")
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared PyTorch CUDA cache")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("gpt-oss-20b model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = None) -> str:
        """Generate response"""
        try:
            # Use the passed parameter, if not specified use default value
            if max_length is None:
                max_length = self.max_new_tokens
            
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Ensure input is on the correct device
            device = next(self.model.parameters()).device
            input_ids = inputs.input_ids.to(device=device, dtype=torch.long)
            attention_mask = inputs.attention_mask.to(device=device, dtype=torch.long) if 'attention_mask' in inputs else None
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    temperature=self.temperature,  # Use the passed temperature parameter
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_beams=1,
                    length_penalty=1.0
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""

def process_single_file(args):
    """Process a single file"""
    file_path, output_dir, model, prompt_template = args
    
    try:
        # Read original transcript
        with open(file_path, 'r', encoding='utf-8') as f:
            original_transcript = f.read().strip()
        
        if not original_transcript:
            return {
                'file': str(file_path),
                'success': False,
                'error': 'Empty transcript'
            }
        
        # Create prompt
        prompt = f"{prompt_template}\n\nTranscript: {original_transcript}\n\nCorrected transcript:"
        
        # Generate correction
        corrected_transcript = model.generate(prompt)
        
        if not corrected_transcript:
            return {
                'file': str(file_path),
                'success': False,
                'error': 'Model returned empty response'
            }
        
        # Save result
        output_path = Path(output_dir) / file_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(corrected_transcript)
        
        return {
            'file': str(file_path),
            'success': True,
            'output_path': str(output_path)
        }
        
    except Exception as e:
        return {
            'file': str(file_path),
            'success': False,
            'error': str(e)
        }

def main():
    """Main function"""
    if len(sys.argv) < 4:
        print("Usage: python llm_gpt_oss_20b.py <input_dir> <output_dir> <prompt> [temperature] [max_new_tokens]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    prompt_template = sys.argv[3]
    
    # Optional temperature and max token parameters
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    max_new_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    
    logger.info(f"Using temperature: {temperature}, max_new_tokens: {max_new_tokens}")
    
    try:
        # Create model
        logger.info("Initializing gpt-oss-20b model...")
        model = GPTOSS20BModel(device="cuda", temperature=temperature, max_new_tokens=max_new_tokens)
        
        # Find transcript files
        input_path = Path(input_dir)
        transcript_files = list(input_path.glob("*.txt"))
        
        if not transcript_files:
            logger.error("No transcript files found")
            sys.exit(1)
        
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        # Process files
        results = []
        successful = 0
        failed = 0
        
        # Use batch processing to improve efficiency
        batch_size = 3  # Small batch processing
        total_batches = (len(transcript_files) + batch_size - 1) // batch_size
        
        with tqdm(total=len(transcript_files), desc="Processing with gpt-oss-20b") as pbar:
            for i in range(0, len(transcript_files), batch_size):
                batch_files = transcript_files[i:i + batch_size]
                batch_args = [
                    (file_path, output_dir, model, prompt_template)
                    for file_path in batch_files
                ]
                
                # Process current batch
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future_to_file = {
                        executor.submit(process_single_file, args): args[0]
                        for args in batch_args
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_file):
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            successful += 1
                        else:
                            failed += 1
                            logger.warning(f"Failed to process {result['file']}: {result.get('error', 'Unknown error')}")
                        
                        pbar.update(1)
        
        # Save summary
        summary = {
            'model': 'gpt-oss-20b',
            'total_files': len(transcript_files),
            'successful': successful,
            'failed': failed,
            'success_rate': f"{(successful/len(transcript_files)*100):.2f}%",
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = Path(output_dir) / "gpt_oss_20b_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing completed:")
        logger.info(f"  Total files: {len(transcript_files)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {summary['success_rate']}")
        logger.info(f"  Summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 