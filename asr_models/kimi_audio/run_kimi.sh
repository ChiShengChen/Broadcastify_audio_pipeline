#!/bin/bash
LD_LIBRARY_PATH=/opt/glibc-2.32/lib:$LD_LIBRARY_PATH \
  /home/meow/anaconda3/envs/kimi/bin/python "/media/meow/One Touch/ems_call/asr_models/kimi_audio/run_transcription.py"
