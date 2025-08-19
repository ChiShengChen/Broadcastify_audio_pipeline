# GPT-OSS-120B 故障排除指南

## 常见错误：'NoneType' object has no attribute 'to_dict'

### 错误描述
```
Error loading model gpt-oss-120b: 'NoneType' object has no attribute 'to_dict'
Failed to create local model: 'NoneType' object has no attribute 'to_dict'
```

### 错误原因
这个错误通常由以下原因引起：

1. **Transformers 库版本不兼容**
   - gpt-oss-120b 需要 transformers >= 4.30.0
   - 较旧版本可能无法正确处理模型配置

2. **模型配置加载问题**
   - 模型配置文件损坏或不完整
   - 网络下载中断导致文件不完整

3. **内存分配问题**
   - 120B 模型需要大量内存
   - CUDA 内存不足时可能导致配置对象为 None

4. **依赖库版本冲突**
   - PyTorch 版本与 transformers 不兼容
   - 其他依赖库版本问题

### 解决方案

#### 方案 1：使用修复版本脚本（推荐）
```bash
# 使用修复版本的脚本
./run_llm_pipeline.sh \
  --asr_results_dir "/path/to/asr/results" \
  --medical_correction_model "gpt-oss-120b" \
  --page_generation_model "gpt-oss-120b"
```

修复版本脚本 (`llm_gpt_oss_120b_fixed.py`) 包含：
- 更好的错误处理
- 版本兼容性检查
- 替代加载方法
- 详细的日志记录

#### 方案 2：升级依赖库
```bash
# 升级 transformers 到兼容版本
pip install --upgrade transformers>=4.30.0

# 升级 PyTorch（如果需要）
pip install --upgrade torch torchvision torchaudio

# 升级其他依赖
pip install --upgrade accelerate bitsandbytes
```

#### 方案 3：使用 gpt-oss-20b 替代
```bash
# 使用更稳定的 20b 模型
./run_llm_pipeline.sh \
  --asr_results_dir "/path/to/asr/results" \
  --medical_correction_model "gpt-oss-20b" \
  --page_generation_model "gpt-oss-20b"
```

#### 方案 4：手动启用 120b 兼容性
```bash
# 如果确定要使用 120b，可以手动启用
./run_llm_pipeline.sh \
  --asr_results_dir "/path/to/asr/results" \
  --medical_correction_model "gpt-oss-120b" \
  --enable_gpt_oss_120b
```

### 预防措施

1. **检查系统要求**
   - 确保有足够的 GPU 内存（建议 80GB+）
   - 检查 CUDA 版本兼容性

2. **验证模型下载**
   ```bash
   # 检查模型文件是否完整
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('openai/gpt-oss-120b')"
   ```

3. **使用量化**
   ```bash
   # 使用 8-bit 或 4-bit 量化减少内存需求
   ./run_llm_pipeline.sh \
     --asr_results_dir "/path/to/asr/results" \
     --medical_correction_model "gpt-oss-120b" \
     --load_in_8bit
   ```

### 调试信息

如果问题仍然存在，可以启用详细日志：

```bash
# 设置环境变量获取更多调试信息
export TRANSFORMERS_VERBOSITY=info
export TORCH_LOGS=all

# 运行脚本
./run_llm_pipeline.sh --asr_results_dir "/path/to/asr/results"
```

### 常见问题解答

**Q: 为什么 20b 模型工作正常，但 120b 失败？**
A: 120b 模型更大更复杂，对依赖库版本和系统资源要求更高。

**Q: 可以同时运行多个 120b 实例吗？**
A: 不建议，除非有足够的 GPU 内存。建议使用批处理模式。

**Q: 修复版本脚本有什么改进？**
A: 包含更好的错误处理、版本检查、替代加载方法和详细日志。

### 联系支持

如果问题仍然存在，请提供以下信息：
1. 完整的错误日志
2. 系统配置（GPU、内存、CUDA 版本）
3. Python 和依赖库版本
4. 使用的脚本参数 