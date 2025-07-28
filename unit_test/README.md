# ASR Pipeline Unit Tests

這個目錄包含了完整的ASR pipeline單元測試套件，用於診斷和復現客戶機上的問題。

## 📋 測試套件概述

### 主要問題診斷
1. **沒有VAD時缺檔問題**：其他模型無法處理所有檔案
2. **VAD後全模型缺檔問題**：所有模型都無法處理所有檔案

### 測試工具
- `create_test_dataset.py` - 創建測試數據集
- `test_pipeline_components.py` - Pipeline組件單元測試
- `vad_diagnosis.py` - VAD專門診斷
- `diagnose_pipeline_limitations.py` - Pipeline限制診斷
- `simple_asr_test.py` - 簡單ASR測試
- `run_comprehensive_tests.py` - 完整測試套件

## 🚀 快速開始

### 1. 運行完整測試套件
```bash
python3 unit_test/run_comprehensive_tests.py
```

這將執行：
- 創建測試數據集
- 運行單元測試
- VAD診斷
- 簡單ASR測試
- Pipeline限制診斷
- 生成綜合報告

### 2. 單獨運行測試

#### 創建測試數據集
```bash
python3 unit_test/create_test_dataset.py
```

#### 運行單元測試
```bash
python3 -m pytest unit_test/test_pipeline_components.py -v
```

#### VAD診斷
```bash
python3 unit_test/vad_diagnosis.py
```

#### Pipeline限制診斷
```bash
python3 unit_test/diagnose_pipeline_limitations.py
```

## 📊 測試數據集

### 測試場景
1. **normal_30s.wav** - 正常語音，30秒，16kHz單聲道
2. **normal_180s.wav** - 長音頻，3分鐘，48kHz立體聲
3. **silence_30s.wav** - 純靜音，30秒
4. **silence_noise.wav** - 低噪音，測試VAD靈敏度
5. **speech_gap.wav** - 語音-靜音-語音模式
6. **intermittent_speech.wav** - 間歇性語音
7. **very_short.wav** - 極短音頻，2秒
8. **very_long.wav** - 極長音頻，5分鐘
9. **low_volume.wav** - 低音量語音
10. **high_noise.wav** - 高噪音語音

### 測試特徵
- 涵蓋各種音頻格式和特性
- 模擬真實使用場景
- 測試邊界條件
- 驗證VAD和ASR處理能力

## 🔍 診斷流程

### 步驟1：環境檢查
```bash
# 檢查系統資源
python3 unit_test/diagnose_pipeline_limitations.py
```

### 步驟2：音頻格式測試
```bash
# 測試音頻格式處理
python3 -m pytest unit_test/test_pipeline_components.py::TestAudioFormat -v
```

### 步驟3：VAD測試
```bash
# 測試VAD處理
python3 -m pytest unit_test/test_pipeline_components.py::TestVADProcessing -v
```

### 步驟4：ASR模型測試
```bash
# 測試ASR模型
python3 -m pytest unit_test/test_pipeline_components.py::TestASRModels -v
```

### 步驟5：Pipeline整合測試
```bash
# 測試Pipeline整合
python3 -m pytest unit_test/test_pipeline_components.py::TestPipelineIntegration -v
```

## 📈 預期結果

### 成功指標
- ✅ 所有測試通過
- ✅ VAD檢測率 > 85%
- ✅ ASR處理成功率 > 90%
- ✅ Pipeline整合成功

### 問題指標
- ❌ VAD檢測率 < 50%
- ❌ ASR處理失敗
- ❌ 音頻格式不兼容
- ❌ 記憶體不足

## 🛠️ 問題診斷

### 常見問題

#### 1. VAD檢測率低
**症狀**：VAD無法檢測到語音片段
**解決方案**：
```bash
# 調整VAD參數
VAD_SPEECH_THRESHOLD=0.2  # 降低閾值
VAD_MIN_SPEECH_DURATION=0.2  # 減少最小語音持續時間
```

#### 2. ASR模型失敗
**症狀**：某些模型無法處理檔案
**解決方案**：
```bash
# 檢查模型可用性
python3 -m pytest unit_test/test_pipeline_components.py::TestASRModels::test_model_availability -v
```

#### 3. 音頻格式問題
**症狀**：音頻檔案無法讀取
**解決方案**：
```bash
# 音頻格式轉換
ffmpeg -i input.wav -ac 1 -ar 16000 output.wav
```

#### 4. 記憶體不足
**症狀**：處理大檔案時崩潰
**解決方案**：
```bash
# 啟用長音頻分割
USE_LONG_AUDIO_SPLIT=true
MAX_SEGMENT_DURATION=120.0
```

## 📝 報告文件

### 生成的報告
- `test_results.json` - 測試結果
- `comprehensive_test_summary.json` - 綜合測試摘要
- `pipeline_limitations_report.json` - Pipeline限制報告
- `fixed_vad_config.json` - 修復的VAD配置

### 報告內容
- 系統資源分析
- 音頻格式限制
- VAD處理限制
- ASR模型限制
- Pipeline整合限制
- 改進建議

## 🔧 自定義測試

### 添加新的測試場景
```python
# 在 create_test_dataset.py 中添加
test_cases.append({
    'name': 'custom_test',
    'duration': 30.0,
    'sample_rate': 16000,
    'channels': 1,
    'type': 'custom_type',
    'description': 'Custom test scenario'
})
```

### 添加新的測試用例
```python
# 在 test_pipeline_components.py 中添加
def test_custom_scenario(self):
    """Test custom scenario"""
    # 測試邏輯
    pass
```

## 📋 依賴項

### 必需套件
```bash
pip install torch torchaudio numpy pandas pytest psutil
```

### 可選套件
```bash
pip install whisper transformers nemo_toolkit[asr]
```

## 🎯 使用建議

### 開發階段
1. 運行完整測試套件
2. 檢查所有報告
3. 根據建議調整配置
4. 重複測試直到通過

### 生產環境
1. 定期運行測試
2. 監控性能指標
3. 根據實際數據調整參數
4. 建立自動化測試流程

## 📞 故障排除

### 常見錯誤

#### 1. 模組導入錯誤
```bash
# 確保在正確的目錄
cd /path/to/ems_call
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### 2. 記憶體不足
```bash
# 減少批次大小
export CUDA_VISIBLE_DEVICES=""  # 使用CPU
```

#### 3. 測試檔案不存在
```bash
# 重新創建測試數據集
python3 unit_test/create_test_dataset.py
```

### 獲取幫助
如果遇到問題，請檢查：
1. 系統資源是否充足
2. 依賴項是否正確安裝
3. 測試檔案是否存在
4. 錯誤日誌中的具體信息

---

**最後更新**：2024年
**版本**：1.0
**維護者**：ASR Pipeline Team 