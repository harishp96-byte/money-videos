"""Hindi voiceover with Indic Parler-TTS (AI4Bharat, Apache-2.0).

Reads episodes/<ep>/script.yaml, speaks every segment's Hindi line and writes:
  out/audio/<segment id>.wav   one file per segment
  out/durations.json           seconds of speech per segment

Runs on a free GitHub Actions CPU machine. Needs the HF_TOKEN secret because
the model is gated on Hugging Face (free account, one-time "agree" click).
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL = "ai4bharat/indic-parler-tts"

# The "desi touch": Parler-TTS takes a plain description of how to speak.
# Edit this line to change the voice (speakers for Hindi: Rohit, Divya).
VOICE = os.environ.get(
    "HINDI_VOICE",
    "Rohit speaks in a warm, friendly and highly expressive Hindi voice, like an "
    "elder brother explaining money to a younger friend, at a moderate pace with "
    "natural pauses. The recording is very clear and close, with no background noise.",
)

GAP = 0.25  # seconds of silence between sentences


def sentences(text: str):
    """Short sentences sound more natural with Parler-TTS."""
    parts = re.split(r"(?<=[।!?.])\s+", text.strip())
    return [p for p in parts if p]


def main(ep_dir: str):
    ep = Path(ep_dir)
    script = yaml.safe_load((ep / "script.yaml").read_text(encoding="utf-8"))
    out = Path("out/audio")
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(os.cpu_count() or 4)
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    rate = model.config.sampling_rate
    desc = desc_tok(VOICE, return_tensors="pt")
    gap = np.zeros(int(GAP * rate), dtype=np.float32)

    durations = {}
    for seg in script["segments"]:
        line = seg.get("say", {}).get("hi", "")
        if not line:
            continue
        pieces = []
        for s in sentences(line):
            print(f"[{seg['id']}] {s}", flush=True)
            p = tok(s, return_tensors="pt")
            with torch.no_grad():
                wav = model.generate(
                    input_ids=desc.input_ids,
                    attention_mask=desc.attention_mask,
                    prompt_input_ids=p.input_ids,
                    prompt_attention_mask=p.attention_mask,
                )
            pieces += [wav.cpu().numpy().squeeze().astype(np.float32), gap]
        audio = np.concatenate(pieces)
        sf.write(out / f"{seg['id']}.wav", audio, rate)
        durations[seg["id"]] = round(len(audio) / rate, 2)

    Path("out/durations.json").write_text(json.dumps(durations, indent=2))
    print("durations:", durations)


if __name__ == "__main__":
    main(sys.argv[1])
