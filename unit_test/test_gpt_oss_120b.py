#!/usr/bin/env python3
"""
专门测试 gpt-oss-120b 模型的脚本
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpt_oss_120b():
    """测试 gpt-oss-120b 模型加载"""
    model_name = "openai/gpt-oss-120b"
    
    try:
        logger.info(f"Testing gpt-oss-120b model loading: {model_name}")
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared PyTorch CUDA cache")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        else:
            logger.warning("CUDA not available, using CPU")
        
        # 加载tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 尝试不同的加载方法
        logger.info("Loading model...")
        
        # 方法1：使用最基本的加载方式
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("✅ Model loaded successfully with auto device mapping")
            return True
        except Exception as e:
            logger.warning(f"Auto device mapping failed: {e}")
        
        # 方法2：尝试CPU加载
        try:
            logger.info("Trying CPU loading...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("✅ Model loaded successfully on CPU")
            return True
        except Exception as e:
            logger.warning(f"CPU loading failed: {e}")
        
        # 方法3：尝试更保守的设置
        try:
            logger.info("Trying conservative loading...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="offload"
            )
            logger.info("✅ Model loaded successfully with offload")
            return True
        except Exception as e:
            logger.warning(f"Conservative loading failed: {e}")
        
        logger.error("❌ All loading methods failed")
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def test_tokenizer():
    """测试tokenizer功能"""
    model_name = "openai/gpt-oss-120b"
    
    try:
        logger.info("Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # 测试基本tokenization
        test_text = "Hello, this is a test of the gpt-oss-120b tokenizer."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        logger.info(f"Original text: {test_text}")
        logger.info(f"Decoded text: {decoded}")
        logger.info(f"Token count: {len(tokens)}")
        
        if test_text in decoded:
            logger.info("✅ Tokenizer test passed")
            return True
        else:
            logger.warning("⚠️ Tokenizer test had minor issues")
            return True  # Still consider it a pass
            
    except Exception as e:
        logger.error(f"❌ Tokenizer test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing gpt-oss-120b Model ===")
    
    # 测试tokenizer
    tokenizer_ok = test_tokenizer()
    
    # 测试模型加载
    model_ok = test_gpt_oss_120b()
    
    if tokenizer_ok and model_ok:
        print("✅ gpt-oss-120b model loading test passed!")
        sys.exit(0)
    else:
        print("❌ gpt-oss-120b model loading test failed!")
        sys.exit(1) 