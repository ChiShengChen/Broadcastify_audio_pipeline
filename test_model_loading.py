#!/usr/bin/env python3
"""
Test Model Loading Script
Tests if the BioMistral-7B model can be loaded successfully with the fixed device handling.
"""

import sys
import logging
from llm_local_models import LocalLLMModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading BioMistral-7B model"""
    try:
        logger.info("Starting model loading test...")
        
        # Test model loading with same parameters as pipeline
        model = LocalLLMModel(
            model_name="BioMistral-7B",
            device="auto",
            load_in_8bit=False,
            load_in_4bit=False
        )
        
        logger.info("✅ Model loaded successfully!")
        
        # Test a simple generation
        test_prompt = "Patient presents with chest pain."
        logger.info("Testing generation with simple prompt...")
        
        result = model.generate(test_prompt)
        if result:
            logger.info("✅ Generation test successful!")
            logger.info(f"Generated text preview: {result[:100]}...")
        else:
            logger.warning("⚠️ Generation returned None")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)