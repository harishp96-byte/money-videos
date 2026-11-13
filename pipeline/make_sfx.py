#!/usr/bin/env python3
"""Generate the ASMR sound effects for the Kayal reel with Stable Audio Open
(github.com/Stability-AI/stable-audio-tools). Runs on CPU in GitHub Actions.

Env: HF_TOKEN (needs the Stable Audio Open licence accepted on Hugging Face),
     STEPS (default 50), ONLY (comma list of names to make), OUT (default audio-library/generated)
Skips files that already exist, so reruns are safe.
"""
import os
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

OUT = os.environ.get("OUT", "audio-library/generated")
STEPS = int(os.environ.get("STEPS", "50"))
ONLY = [x.strip() for x in os.environ.get("ONLY", "").split(",") if x.strip()]
SECONDS = 8

SFX = {
    "village_dawn": "village morning ambience, birds chirping, soft wind, distant footsteps on dirt, close-mic ASMR",
    "workshop_room": "quiet pottery workshop room tone, gentle clay pots being touched, soft wooden shelf, ASMR",
    "dig_clay": "spade digging into wet red clay, heavy thunk and squelch, close-mic foley ASMR",
    "sift_grain": "sifting dry soil through a mesh sieve, soft grainy shush, small pebbles clinking, ASMR",
    "knead_clay": "kneading wet clay by hand, squishy slow presses, water bubbles, close-mic ASMR",
    "pottery_wheel": "pottery wheel spinning slowly, low hum, wet hands smoothing clay, ASMR",
    "pot_tap_kiln": "tapping a fired clay pot with a knuckle, ceramic ring, wood kiln fire crackling, ASMR",
    "pongal_boil": "milk boiling and bubbling in a clay pot, foam rising, overflow, temple bell ringing",
    "footsteps_dirt": "bare feet running on dry dirt path, anklet bells jingling, ASMR",
    "temple_bell": "small brass temple bell ringing, soft reverb, morning",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device)
sr, size = config["sample_rate"], config["sample_size"]
os.makedirs(OUT, exist_ok=True)

for name, prompt in SFX.items():
    if ONLY and name not in ONLY:
        continue
    path = f"{OUT}/{name}.wav"
    if os.path.exists(path):
        print("skip", name)
        continue
    cond = [{"prompt": prompt, "seconds_start": 0, "seconds_total": SECONDS}]
    with torch.inference_mode():
        out = generate_diffusion_cond(model, steps=STEPS, cfg_scale=7, conditioning=cond,
                                      sample_size=size, sigma_min=0.3, sigma_max=500,
                                      sampler_type="dpmpp-3m-sde", device=device)
    out = rearrange(out, "b d n -> d (b n)").to(torch.float32).cpu()
    out = out[:, : SECONDS * sr]
    out = out / max(out.abs().max().item(), 1e-6) * 0.9
    torchaudio.save(path, out, sr)
    print("wrote", path)

with open(f"{OUT}/PROMPTS.md", "w") as f:
    f.write("Generated with Stable Audio Open 1.0 (Stability AI Community Licence). Check terms before monetised use.\n\n")
    for n, p in SFX.items():
        f.write(f"- {n}.wav: {p}\n")
