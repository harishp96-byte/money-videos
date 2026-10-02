#!/usr/bin/env python3
"""tts_hindi.py: speak each segment's Hindi line with Indic Parler-TTS."""
import os
import re
import json
from pathlib import Path

import yaml
import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

MODEL = "ai4bharat/indic-parler-tts"
DEFAULT_VOICE = (
    "Rohit speaks in a warm, friendly and highly expressive Hindi voice, "
    "like an elder brother explaining money to a younger friend, at a moderate "
    "pace with natural pauses. The recording is very clear and close, "
    "with no background noise."
)


def main():
    episode_dir = Path(os.environ.get("EPISODE_DIR", "."))
    voice = os.environ.get("VOICE", DEFAULT_VOICE)
    out_dir = episode_dir / "out"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with open(episode_dir / "script.yaml", encoding="utf-8") as f:
        script = yaml.safe_load(f)
    segments = script["segments"] if isinstance(script, dict) else script

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate

    desc_ids = desc_tok(voice, return_tensors="pt").to(device)
    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    durations = {}

    for seg in segments:
        text = (seg.get("say") or {}).get("hi", "").strip()
        if not text:
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[।!?.])\s*", text) if s.strip()]
        pieces = []
        for s in sentences:
            prompt = tok(s, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=desc_ids.input_ids,
                    attention_mask=desc_ids.attention_mask,
                    prompt_input_ids=prompt.input_ids,
                    prompt_attention_mask=prompt.attention_mask,
                )
            pieces.append(gen.cpu().numpy().squeeze().astype(np.float32))
            pieces.append(gap)
        audio = np.concatenate(pieces)
        sf.write(str(audio_dir / f"{seg['id']}.wav"), audio, sr)
        durations[seg["id"]] = round(len(audio) / sr, 2)
        print(f"spoke {seg['id']}: {durations[seg['id']]}s")

    with open(out_dir / "durations.json", "w") as f:
        json.dump(durations, f, indent=2)


if __name__ == "__main__":
    main()
