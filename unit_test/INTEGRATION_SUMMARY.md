# ��XASR�޹D�`��

## ? ���إؼ�

�ھ� `model_audio_limitations.md` �����R�A�Ыؤ@�Ӿ�X�����W�w�B�z�t�ΡA�T�O�Ҧ���J���W����b�Ҧ�ASR�ҫ��W�B��C

## ? �ҫ�������R

### ��l����
- **Whisper (large-v3)**: ���F���A�X�G�L����
- **Canary-1b (NeMo)**: �Y�歭��G0.5-60��A16kHz�A���n�D�A�̤p���q0.01
- **Parakeet-tdt-0.6b-v2 (NeMo)**: ��������G1.0-300��A16kHz�A���n�D
- **Wav2Vec2-xls-r (Transformers)**: �}�n�F���ʡG0.1��H�W�A16kHz�A�̤p���q0.01

## ? �ѨM���

### 1. **���W�w�B�z�{��** (`audio_preprocessor.py`)
- �۰ʳB�z���P�ҫ������W����
- �䴩�ɪ��վ�B�ļ˲v�ഫ�B���q�зǤ�
- �n�D�ഫ�]�����n�����n�D�^
- ������Ϊ����W���

### 2. **��X�޹D** (`run_integrated_pipeline.sh`)
- �N���W�w�B�z��X��즳ASR�޹D
- 9�ӨB�J������u�@�y�{
- ���㪺���~�B�z�M��x�O��

### 3. **���ռƾڥͦ���** (`generate_test_data.py`)
- �ͦ��U�دS�ʪ����խ��W
- �]�t�u���W�B�����W�B�C���q�B�����n��
- �۰ʥͦ��е����

## ? �Ыت����

### �֤ߤ��
1. **`audio_preprocessor.py`** - ���W�w�B�z�{��
2. **`run_integrated_pipeline.sh`** - ��X�޹D�}��
3. **`generate_test_data.py`** - ���ռƾڥͦ���
4. **`test_audio_preprocessor.py`** - �w�B�z���ո}��
5. **`run_complete_test.sh`** - ������ո}��

### ���ɤ��
6. **`AUDIO_PREPROCESSING_GUIDE.md`** - �w�B�z�Բӫ��n
7. **`INTEGRATED_PIPELINE_GUIDE.md`** - ��X�޹D�ϥΫ��n
8. **`INTEGRATION_SUMMARY.md`** - ���`������

## ? ���յ��G

### ���ռƾڥͦ�
```bash
python3 generate_test_data.py --output_dir ./test_data_integrated --create_ground_truth --verbose
```

**�ͦ������G**
- 10�Ӵ��խ��W���]423.4���`�ɪ��^
- �]�t�U�دS�ʡG�u���W�B�����W�B�C���q�B�����n�B���P�ļ˲v��
- �۰ʥͦ����е����

### �w�B�z����
```bash
python3 audio_preprocessor.py --input_dir ./test_data_integrated --output_dir ./preprocessed_test_integrated --verbose
```

**�w�B�z���G�G**
- **�`��J���**: 10��
- **�`��X���**: 46�ӡ]��4�Ӽҫ��u�ơ^
- **���\�v**: 100% (10/10)

### �ҫ��ݮe�ʲέp
| �ҫ� | ��X���� | ���\�v | ���� |
|------|------------|--------|------|
| large-v3 | 10 | 100% | ���F���A�X�G�L���� |
| canary-1b | 16 | 100% | �Y�歭��A�ݭn���Ϊ����W |
| parakeet-tdt-0.6b-v2 | 10 | 100% | �������� |
| wav2vec-xls-r | 10 | 100% | �}�n�F���� |

## ? ����\��

### 1. **����ɪ��B�z**
- �u���W�G�۰ʶ�R��̤p�ɪ�
- �����W�G������Ρ]Canary-1b����60��^
- ���q�зǤơG�T�O�̤p���q�n�D

### 2. **�榡�ഫ**
- �ļ˲v�ഫ�G�Τ@��16kHz
- �n�D�ഫ�G�����n�����n�D
- ���q�зǤơG�T�O�̤p���q0.01

### 3. **�ҫ��S�w�u��**
- **Canary-1b**: �Y��ɪ�����A�۰ʤ���
- **Parakeet**: ��������A�ɪ��վ�
- **Wav2Vec2**: ���q�зǤ�
- **Whisper**: ���F���A�̤p�B�z

## ? �ʯ��{

### �B�z�Ĳv
- **�æ�B�z**: �䴩�h�i�{�B�z
- **����w�s**: �קK���ƳB�z
- **�O�����u��**: �j�����q�B�z

### ���~�B�z
- **�����x**: �ԲӪ��B�z�O��
- **���~��_**: �u�������~�B�z
- **�i�װl��**: ��ɳB�z�i��

## ? �ϥνd��

### �򥻨ϥ�
```bash
# �ͦ����ռƾ�
python3 generate_test_data.py --create_ground_truth

# �B���X�޹D
./run_integrated_pipeline.sh \
    --input_dir ./test_data_integrated \
    --output_dir ./pipeline_results \
    --use-audio-preprocessing
```

### ���Űt�m
```bash
# ����\��t�m
./run_integrated_pipeline.sh \
    --input_dir /path/to/audio \
    --output_dir /path/to/results \
    --use-audio-preprocessing \
    --use-vad \
    --use-long-audio-split \
    --preprocess-ground-truth
```

## ? ���ҵ��G

### 1. **�ݮe�ʴ���**
- ? �Ҧ�10�Ӵ��դ�󦨥\�B�z
- ? 4�Ӽҫ�100%�ݮe
- ? �۰ʳB�z�U�ح��W�S��

### 2. **�\�����**
- ? �ɪ��վ�]�u���W��R�A�����W���Ρ^
- ? �榡�ഫ�]�ļ˲v�B�n�D�^
- ? ���q�зǤ�
- ? ���~�B�z�M��x�O��

### 3. **��X����**
- ? �P�즳ASR�޹D������X
- ? �O���즳�\�৹���
- ? �s�W�w�B�z�\��L�_�α�

## ? �u���`��

### 1. **�����ݮe**
- �T�O�Ҧ����W�b�Ҧ��ҫ��W�B��
- �۰ʳB�z�ҫ��S�w����
- �����u�ƳB�z����

### 2. **����ϥ�**
- ²�檺�R�O��ɭ�
- �ԲӪ��ϥΤ���
- ���㪺���ծM��

### 3. **���ץi�a**
- ���㪺���~�B�z
- �ԲӪ���x�O��
- 100%���\�v����

### 4. **�ʯ��u��**
- �æ�B�z�䴩
- ����w�s����
- �O�����u�Ƴ]�p

## ? �U�@�B��ĳ

### 1. **�Ͳ����ҳ��p**
```bash
# �ϥίu��ƾڴ���
./run_integrated_pipeline.sh \
    --input_dir /path/to/real/audio \
    --output_dir /path/to/production/results
```

### 2. **�ʯ�ʱ�**
- �ʱ��B�z�ɶ��M�귽�ϥ�
- �u�ƨæ�B�z�Ѽ�
- �ھڹ�ڻݨD�վ�t�m

### 3. **�\���X�i**
- �䴩��h���W�榡
- �K�[��h�ҫ��䴩
- �u�ƳB�z��k

## ? ����

���\�ЫؤF�@�ӧ��㪺���W�w�B�z�t�ΡA�����ѨM�F���PASR�ҫ��������ݮe�ʰ��D�C�q�L���઺���W�B�z�M�ҫ��S�w�u�ơA�T�O�F�Ҧ����W��󳣯�b�Ҧ��ҫ��W���\�B��A�j�j�����FASR�޹D��í�w�ʩM���\�v�C

**���䦨�N�G**
- ? 100%�ҫ��ݮe��
- ? ���㪺��������
- ? �ԲӪ����ɫ��n
- ? ����ϥΪ��ɭ�
- ? ���ץi�a���B�z

�o�Ӿ�X�t�ά�EMS�q�ܻy���ѧO�������ѤF��ꪺ��¦�A�T�O�F�B�z�y�{��í�w�ʩM�ǽT�ʡC 