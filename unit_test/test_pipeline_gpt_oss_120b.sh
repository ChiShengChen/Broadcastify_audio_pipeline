#!/bin/bash

# 测试 gpt-oss-120b 在 pipeline 中的集成

echo "=== Testing gpt-oss-120b Integration with Custom Parameters ==="

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行 pipeline 测试，使用自定义参数
./run_llm_pipeline.sh \
    --asr_results_dir "/media/meow/One Touch/ems_call/pipeline_results_20250814_044143" \
    --output_dir "/media/meow/One Touch/ems_call/test_pipeline_gpt_oss_120b" \
    --medical_correction_model "gpt-oss-120b" \
    --page_generation_model "gpt-oss-120b" \
    --batch_size 1 \
    --load_in_4bit \
    --temperature 0.2 \
    --max_new_tokens 256

echo "=== Test completed ===" 