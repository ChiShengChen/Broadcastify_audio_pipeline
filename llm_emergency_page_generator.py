#!/usr/bin/env python3
"""
LLM Emergency Page Generator Script

This script uses LLM models to generate structured emergency pages from medical transcripts.
Supports multiple LLM models including local and OpenAI-compatible APIs.
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
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_emergency_page_generator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LLMClient:
    """Base class for LLM API clients"""
    
    def __init__(self, model: str, endpoint: str, api_key: str, timeout: int = 60):
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate response from LLM"""
        raise NotImplementedError
    
    def _make_request(self, url: str, data: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, json=data, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    return None
        return None

class OpenAICompatibleClient(LLMClient):
    """Client for OpenAI-compatible APIs (GPT-OSS, etc.)"""
    
    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate response using OpenAI-compatible API"""
        url = f"{self.endpoint}/chat/completions"
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an emergency medical dispatcher."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "stream": False
        }
        
        response = self._make_request(url, data, max_retries)
        if response and 'choices' in response:
            return response['choices'][0]['message']['content'].strip()
        return None

class LocalModelClient(LLMClient):
    """Client for local model APIs (BioMistral, Meditron, Llama)"""
    
    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate response using local model API"""
        url = f"{self.endpoint}/chat/completions"
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an emergency medical dispatcher."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "stream": False
        }
        
        response = self._make_request(url, data, max_retries)
        if response and 'choices' in response:
            return response['choices'][0]['message']['content'].strip()
        return None

def create_llm_client(model: str, local_endpoint: str, openai_base: str) -> LLMClient:
    """Create appropriate LLM client based on model type"""
    
    # Models that use OpenAI-compatible API
    openai_models = ["gpt-oss-20b", "gpt-oss-120b"]
    
    # Models that use local API
    local_models = ["BioMistral-7B", "Meditron-7B", "Llama-3-8B-UltraMedica"]
    
    if model in openai_models:
        return OpenAICompatibleClient(model, openai_base, "local")
    elif model in local_models:
        return LocalModelClient(model, local_endpoint, "local")
    else:
        raise ValueError(f"Unknown model: {model}")

def load_transcript(file_path: Path) -> str:
    """Load transcript from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error loading transcript {file_path}: {e}")
        return ""

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

def generate_emergency_page(transcript: str, client: LLMClient, prompt_template: str) -> Optional[str]:
    """Generate emergency page from transcript using LLM"""
    try:
        # Create prompt with transcript
        prompt = f"{prompt_template}\n\nMedical Transcript: {transcript}\n\nEmergency Page:"
        
        # Generate emergency page
        emergency_page = client.generate(prompt)
        if emergency_page:
            return emergency_page.strip()
        else:
            logger.warning("LLM returned empty response")
            return None
    except Exception as e:
        logger.error(f"Error generating emergency page: {e}")
        return None

def process_single_file(args: tuple) -> Dict[str, Any]:
    """Process a single transcript file"""
    file_path, output_dir, client, prompt_template = args
    
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
        emergency_page = generate_emergency_page(transcript, client, prompt_template)
        if emergency_page is None:
            return {
                'file': str(file_path),
                'success': False,
                'error': 'LLM page generation failed'
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

def create_emergency_page_template() -> str:
    """Create a structured emergency page template"""
    return """EMERGENCY PAGE TEMPLATE

PATIENT CONDITION SUMMARY:
[Brief description of patient's condition and symptoms]

LOCATION DETAILS:
- Address: [Specific address or location]
- Landmarks: [Nearby landmarks or directions]
- Access: [Accessibility information]

REQUIRED MEDICAL RESOURCES:
- Ambulance Type: [Basic/Advanced Life Support]
- Special Equipment: [Any special equipment needed]
- Additional Units: [Fire, police, etc. if needed]

PRIORITY LEVEL:
- Priority: [High/Medium/Low]
- Response Time: [Immediate/Urgent/Routine]

KEY MEDICAL INFORMATION:
- Chief Complaint: [Main reason for call]
- Vital Signs: [If available]
- Allergies: [If known]
- Medications: [If relevant]
- Medical History: [If relevant]

DISPATCHER NOTES:
[Additional information or special instructions]"""

def main():
    parser = argparse.ArgumentParser(description="LLM Emergency Page Generator")
    parser.add_argument("--input_dirs", nargs="+", required=True,
                       help="Input directories containing transcript files")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for emergency pages")
    parser.add_argument("--model", required=True,
                       choices=["gpt-oss-20b", "gpt-oss-120b", "BioMistral-7B", "Meditron-7B", "Llama-3-8B-UltraMedica"],
                       help="LLM model to use for page generation")
    parser.add_argument("--local_endpoint", default="http://localhost:8000/v1",
                       help="Local model API endpoint")
    parser.add_argument("--openai_base", default="http://localhost:8000/v1",
                       help="OpenAI-compatible API base URL")
    parser.add_argument("--batch_size", type=int, default=5,
                       help="Number of files to process in parallel")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum retry attempts for API calls")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Timeout for API requests in seconds")
    parser.add_argument("--prompt", default="You are an emergency medical dispatcher. Based on the following corrected medical transcript, generate a structured emergency page that includes: 1) Patient condition summary, 2) Location details, 3) Required medical resources, 4) Priority level, 5) Key medical information. Format the response as a structured emergency page.",
                       help="Custom prompt for emergency page generation")
    parser.add_argument("--use_template", action="store_true",
                       help="Use structured template for emergency pages")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create LLM client
    try:
        client = create_llm_client(args.model, args.local_endpoint, args.openai_base)
        logger.info(f"Created LLM client for model: {args.model}")
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        sys.exit(1)
    
    # Find transcript files
    transcript_files = find_transcript_files(args.input_dirs)
    if not transcript_files:
        logger.error("No transcript files found in input directories")
        sys.exit(1)
    
    logger.info(f"Found {len(transcript_files)} transcript files to process")
    
    # Prepare prompt
    if args.use_template:
        prompt_template = f"{args.prompt}\n\nUse the following template structure:\n\n{create_emergency_page_template()}"
    else:
        prompt_template = args.prompt
    
    # Process files
    results = []
    successful = 0
    failed = 0
    
    # Prepare arguments for parallel processing
    process_args = [
        (file_path, output_dir, client, prompt_template)
        for file_path in transcript_files
    ]
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, args): args[0]
            for args in process_args
        }
        
        # Process results as they complete
        with tqdm(total=len(transcript_files), desc="Generating emergency pages") as pbar:
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
        'model': args.model,
        'input_directories': args.input_dirs,
        'output_directory': str(output_dir),
        'total_files': len(transcript_files),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(transcript_files) if transcript_files else 0,
        'use_template': args.use_template,
        'results': results
    }
    
    summary_file = output_dir / "emergency_page_generation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save template for reference
    if args.use_template:
        template_file = output_dir / "emergency_page_template.txt"
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(create_emergency_page_template())
    
    # Log final results
    logger.info(f"Processing completed:")
    logger.info(f"  Total files: {len(transcript_files)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Success rate: {summary['success_rate']:.2%}")
    logger.info(f"  Summary saved to: {summary_file}")
    
    if args.use_template:
        logger.info(f"  Template saved to: {template_file}")
    
    if failed > 0:
        logger.warning(f"{failed} files failed to process. Check the summary for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 