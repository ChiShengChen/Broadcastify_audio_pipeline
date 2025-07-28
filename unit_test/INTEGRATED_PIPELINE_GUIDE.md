# ��XASR�޹D�ϥΫ��n

## ���z

�o�Ӿ�X��ASR�޹D�N���W�w�B�z�\��P�즳��ASR�����޹D�������X�A�T�O�Ҧ����W��󳣯�b�Ҧ�ASR�ҫ��W�B��C

## ? �D�n�S��

### 1. **���W�w�B�z��X**
- �۰ʳB�z���P�ҫ������W����
- �䴩�ɪ��վ�B�ļ˲v�ഫ�B���q�зǤ�
- �T�O�Ҧ����W�ŦX�U�ҫ��n�D

### 2. **����u�@�y�{**
- ���W�w�B�z �� VAD�B�z �� ASR��� �� ����
- �䴩�����W���ΩM�е��w�B�z
- ���㪺���~�B�z�M��x�O��

### 3. **�ҫ��ݮe��**
- **Whisper (large-v3)**: ���F���A�X�G�L����
- **Canary-1b**: 0.5-60��A16kHz�A���n�D
- **Parakeet-tdt-0.6b-v2**: 1.0-300��A16kHz�A���n�D
- **Wav2Vec2-xls-r**: 0.1��H�W�A16kHz�A���n�D

## ? ��󵲺c

```
ems_call/
�u�w�w run_integrated_pipeline.sh      # ��X�޹D�D�}��
�u�w�w audio_preprocessor.py           # ���W�w�B�z�{��
�u�w�w generate_test_data.py           # ���ռƾڥͦ���
�u�w�w test_audio_preprocessor.py      # �w�B�z���ո}��
�u�w�w run_complete_test.sh            # ������ո}��
�u�w�w AUDIO_PREPROCESSING_GUIDE.md   # �w�B�z�Բӫ��n
�|�w�w INTEGRATED_PIPELINE_GUIDE.md   # �����n
```

## ? �ֳt�}�l

### 1. **�ͦ����ռƾ�**
```bash
# �ͦ����խ��W���M�е�
python3 generate_test_data.py \
    --output_dir ./test_data \
    --create_ground_truth \
    --verbose
```

### 2. **�B�槹�����**
```bash
# �B�槹�㪺���ծM��
./run_complete_test.sh
```

### 3. **�ϥξ�X�޹D**
```bash
# �򥻨ϥ�
./run_integrated_pipeline.sh \
    --input_dir ./test_data \
    --output_dir ./pipeline_results

# ���ſﶵ
./run_integrated_pipeline.sh \
    --input_dir ./test_data \
    --output_dir ./pipeline_results \
    --use-audio-preprocessing \
    --use-vad \
    --use-long-audio-split \
    --preprocess-ground-truth
```

## ? �t�m�ﶵ

### ���W�w�B�z�ﶵ
```bash
--use-audio-preprocessing          # �ҥέ��W�w�B�z
--no-audio-preprocessing           # �T�έ��W�w�B�z
```

### VAD�ﶵ
```bash
--use-vad                         # �ҥ�VAD�B�z
--vad-threshold FLOAT             # �y���˴��H�� (�w�]: 0.5)
--vad-min-speech FLOAT            # �̤p�y������ɶ� (�w�]: 0.5s)
--vad-min-silence FLOAT           # �̤p�R������ɶ� (�w�]: 0.3s)
```

### �����W���οﶵ
```bash
--use-long-audio-split            # �ҥΪ����W����
--max-segment-duration FLOAT      # �̤j���q����ɶ� (�w�]: 120.0s)
```

### �е��w�B�z�ﶵ
```bash
--preprocess-ground-truth          # �ҥμе��w�B�z
--no-preprocess-ground-truth       # �T�μе��w�B�z
--preprocess-mode MODE             # �w�B�z�Ҧ� (conservative/aggressive)
```

## ? ��X���c

```
pipeline_results_YYYYMMDD_HHMMSS/
�u�w�w preprocessed_audio/            # �w�B�z�᪺���W���
�x   �u�w�w audio1_large-v3.wav
�x   �u�w�w audio1_canary-1b.wav
�x   �u�w�w audio1_parakeet-tdt-0.6b-v2.wav
�x   �|�w�w audio1_wav2vec-xls-r.wav
�u�w�w long_audio_segments/           # �����W���ε��G
�u�w�w vad_segments/                  # VAD�B�z���G
�u�w�w asr_transcripts/               # ASR������G
�u�w�w merged_transcripts/            # �X�֪�������G
�u�w�w asr_evaluation_results.csv     # �������G
�u�w�w model_file_analysis.txt        # �ҫ������R
�u�w�w integration_summary.txt        # ��X�K�n
�|�w�w error.log                      # ���~��x
```

## ? ���ե\��

### 1. **�ֳt����**
```bash
# �ˬd�̿ඵ�ùB��򥻴���
./quick_start.sh
```

### 2. **�w�B�z����**
```bash
# ���խ��W�w�B�z�\��
python3 test_audio_preprocessor.py
```

### 3. **�������**
```bash
# �B�槹�㪺���ծM��
./run_complete_test.sh
```

### 4. **�۩w�q����**
```bash
# �ͦ��۩w�q���ռƾ�
python3 generate_test_data.py \
    --output_dir ./my_test_data \
    --num_files 5 \
    --create_ground_truth \
    --verbose

# ���չw�B�z
python3 audio_preprocessor.py \
    --input_dir ./my_test_data \
    --output_dir ./my_preprocessed \
    --verbose
```

## ? �ʯ��u��

### 1. **��q�B�z**
```bash
# �@���B�z�h�ӭ��W���
./run_integrated_pipeline.sh \
    --input_dir /path/to/large/audio/dataset \
    --output_dir /path/to/results
```

### 2. **�æ�B�z**
```python
# �b audio_preprocessor.py ���i�H�ҥΦh�i�{
# �ק� num_workers �Ѽ�
```

### 3. **�O�����u��**
```bash
# �ϥΪ����W�����קKOOM
./run_integrated_pipeline.sh \
    --use-long-audio-split \
    --max-segment-duration 60
```

## ? �G�ٱư�

### 1. **�`�����D**

**���D�G���W���L�kŪ��**
```bash
# �ѨM��סG�ˬd���榡
file audio_file.wav
# �T�O��󥼷l�a�B�榡���T
```

**���D�G�O���餣��**
```bash
# �ѨM��סG�ҥΪ����W����
./run_integrated_pipeline.sh \
    --use-long-audio-split \
    --max-segment-duration 60
```

**���D�G�w�B�z����**
```bash
# �ѨM��סG�ˬd�̿ඵ
python3 -c "import soundfile, librosa, numpy; print('Dependencies OK')"
```

### 2. **�ոէޥ�**

```bash
# �ҥθԲӤ�x
./run_integrated_pipeline.sh \
    --input_dir ./test_data \
    --output_dir ./debug_results \
    --verbose

# �ˬd���W�H��
python3 -c "
import soundfile as sf
info = sf.info('audio.wav')
print(f'Duration: {info.duration}s')
print(f'Sample rate: {info.samplerate}Hz')
print(f'Channels: {info.channels}')
"
```

## ? �ϥνd��

### �d��1�G�򥻨ϥ�
```bash
# �ϥιw�]�t�m�B��޹D
./run_integrated_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv
```

### �d��2�G���Űt�m
```bash
# �ϥΩҦ��\��
./run_integrated_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --ground_truth /path/to/ground_truth.csv \
    --use-audio-preprocessing \
    --use-vad \
    --vad-threshold 0.6 \
    --use-long-audio-split \
    --max-segment-duration 90 \
    --preprocess-ground-truth \
    --preprocess-mode aggressive
```

### �d��3�G���ռҦ�
```bash
# �ͦ����ռƾڨùB��
python3 generate_test_data.py --create_ground_truth
./run_integrated_pipeline.sh \
    --input_dir ./test_data \
    --output_dir ./test_results
```

## ? �ʱ��M���i

### 1. **�i�׺ʱ�**
```bash
# �d�ݳB�z�i��
tail -f pipeline_results_*/integration_summary.txt
```

### 2. **���G���R**
```bash
# �d�ݵ������G
cat pipeline_results_*/asr_evaluation_results.csv

# �d�ݼҫ����R
cat pipeline_results_*/model_file_analysis.txt
```

### 3. **���~�ˬd**
```bash
# �ˬd���~��x
cat pipeline_results_*/error.log

# �ˬdĵ�i
grep "WARNING" pipeline_results_*/error.log
```

## ? �̨ι��

### 1. **�ƾڷǳ�**
- �T�O���W���榡���T�]WAV, MP3, M4A, FLAC�^
- �ˬd�е����榡�]CSV with Filename, transcript columns�^
- ���Ҥ����|�M�v��

### 2. **�t�έn�D**
- Python 3.7+
- �������ϺЪŶ��]�w�B�z�|�ͦ��B�~���^
- ��ĳ8GB+ RAM�Ω�j���B�z

### 3. **�ʯ��u��**
- �ϥ�SSD�s�x�H����I/O�ʯ�
- �ھڨt�ΰt�m�վ�æ�B�z�Ѽ�
- �w���M�z�{�ɤ��

## ? ��s�M���@

### 1. **�ˬd��s**
```bash
# �ˬd�}�������M�̿ඵ
python3 -c "import soundfile, librosa, numpy; print('All dependencies up to date')"
```

### 2. **�ƥ��t�m**
```bash
# �ƥ����n�t�m
cp run_integrated_pipeline.sh run_integrated_pipeline.sh.backup
```

### 3. **�M�z�¤��**
```bash
# �M�z�ª����թM���G���
rm -rf test_data_* pipeline_results_* preprocessed_*
```

## ? �䴩

�p�G�J����D�A���ˬd�G

1. **�̿ඵ�w��**
```bash
pip install soundfile librosa numpy scipy
```

2. **����v��**
```bash
chmod +x run_integrated_pipeline.sh
```

3. **Python���|**
```bash
python3 --version
which python3
```

4. **�t�θ귽**
```bash
df -h  # �ˬd�ϺЪŶ�
free -h  # �ˬd�O����
```

## ? �`��

�o�Ӿ�X��ASR�޹D���ѤF�G

- ? **���㪺���W�w�B�z**�G�T�O�Ҧ����W�ŦX�ҫ��n�D
- ? **�F�����t�m�ﶵ**�G�䴩�U�بϥγ���
- ? **���㪺���ծM��**�G�T�O�\�ॿ�T��
- ? **�ԲӪ����~�B�z**�G���ѲM�������~�H��
- ? **����������**�G�]�t�ϥΫ��n�M�̨ι��

�q�L�o�Ӿ�X�޹D�A�z�i�H�T�O�Ҧ����W��󳣯�b�Ҧ�ASR�ҫ��W���\�B��A�j�j�����F�޹D��í�w�ʩM���\�v�I 