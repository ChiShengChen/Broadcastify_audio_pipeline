"""Run a Whisper model (vanilla or LoRA-fine-tuned) on a HuggingFace Dataset
and write per-clip predictions to CSV.

Output CSV columns: index, original_file, segment_index, reference, prediction
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Some envs break on transformers->accelerate->boto3->urllib3 mismatch.
# If real boto3 isn't already importable, stub it out before transformers loads.
# The stub needs a real ModuleSpec so accelerate's importlib.util.find_spec
# doesn't trip on __spec__=None (Python 3.10+ behavior).
import types
if "boto3" not in sys.modules:
    try:
        import boto3 as _real_boto3  # noqa: F401
    except Exception:
        from importlib.machinery import ModuleSpec
        _b = types.ModuleType("boto3"); _bs = types.ModuleType("boto3.session")
        _b.__spec__ = ModuleSpec("boto3", loader=None)
        _bs.__spec__ = ModuleSpec("boto3.session", loader=None)
        _bs.Session = type("DummySession", (), {})
        _b.session = _bs
        sys.modules["boto3"] = _b
        sys.modules["boto3.session"] = _bs

import torch
import pandas as pd
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")

try:
    from peft import PeftModel
    PEFT_OK = True
except ImportError:
    PEFT_OK = False


def load_model(model_path: str, device: str):
    """Load a Whisper model — supports plain HF model id or a LoRA adapter dir."""
    is_lora = os.path.isdir(model_path) and os.path.exists(
        os.path.join(model_path, "adapter_config.json"))

    if is_lora:
        if not PEFT_OK:
            raise ImportError("peft is required to load LoRA adapter; pip install peft")
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            base_name = json.load(f).get("base_model_name_or_path",
                                         "openai/whisper-large-v3")
        print(f"  LoRA adapter detected; base model: {base_name}")
        processor = WhisperProcessor.from_pretrained(base_name)
        base = WhisperForConditionalGeneration.from_pretrained(base_name)
        model = PeftModel.from_pretrained(base, model_path).merge_and_unload()
    else:
        print(f"  loading full model: {model_path}")
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)

    model.eval().to(device)
    if device == "cuda":
        model = model.half()
    return processor, model


def evaluate(model_path: str, dataset_dir: str, output_csv: str,
             device: str = None) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(model_path, device)
    dtype = next(model.parameters()).dtype

    ds = load_from_disk(dataset_dir)
    print(f"  evaluating {len(ds)} clips on {device} ({dtype})")

    rows = []
    with torch.no_grad():
        for idx, ex in enumerate(tqdm(ds, desc=f"  → {Path(output_csv).stem}")):
            a = ex["audio"]["array"]
            sr = ex["audio"]["sampling_rate"]
            feat = processor.feature_extractor(a, sampling_rate=sr,
                                               return_tensors="pt").input_features.to(device)
            if dtype == torch.float16:
                feat = feat.half()
            ids = model.generate(
                feat, language="en", task="transcribe",
                no_repeat_ngram_size=3, repetition_penalty=1.2,
            )
            pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
            rows.append({
                "index": idx,
                "original_file": ex.get("original_file", "unknown"),
                "segment_index": ex.get("segment_index", 0),
                "reference": ex["transcript"],
                "prediction": pred,
            })

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"  saved → {output_csv}")
    # Free GPU memory before next model
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    return output_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="HF model id (e.g. openai/whisper-large-v3) or LoRA adapter dir")
    ap.add_argument("--dataset_dir", required=True,
                    help="HuggingFace Dataset dir produced by prepare_dataset.py")
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()
    evaluate(args.model_path, args.dataset_dir, args.output_csv)
