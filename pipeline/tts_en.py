"""English voiceover with Kokoro-82M (Apache-2.0, free, runs on CPU).

Reads episodes/<ep>/script.yaml, speaks every segment's English line in a
young, cheerful voice and writes:
  out/audio/<segment id>.wav   one file per segment
  out/durations.json           seconds of speech per segment (sets scene length)
  out/voice.json               per segment: word timings (for captions) and a
                               loudness curve at 30 fps (for mouth movement)
  out/voice_lang.txt           "en"

Voice settings can be put in script.yaml:
  voice_en: {voice: af_heart, speed: 1.0, pitch: 1.12}
pitch > 1 raises the voice (1.12 makes it sound like a young boy) while
keeping the speed the same.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from kokoro import KPipeline

RATE = 24000
FPS = 30
GAP = 0.18      # pause between sentences


def main(ep_dir):
    ep = Path(ep_dir)
    script = yaml.safe_load((ep / "script.yaml").read_text(encoding="utf-8"))
    cfg = {"voice": "af_heart", "speed": 1.0, "pitch": 1.12, **(script.get("voice_en") or {})}
    out = Path("out/audio")
    out.mkdir(parents=True, exist_ok=True)
    pipe = KPipeline(lang_code="a")
    durations, info = {}, {}

    for seg in script["segments"]:
        text = (seg.get("say") or {}).get("en")
        if not text:
            continue
        chunks, words, t0 = [], [], 0.0
        for r in pipe(text, voice=cfg["voice"], speed=float(cfg["speed"]), split_pattern=r"(?<=[.!?])\s+"):
            audio = r.audio.detach().cpu().numpy() if hasattr(r.audio, "detach") else np.asarray(r.audio)
            for tok in (r.tokens or []):
                if not tok.text.strip():
                    continue
                # glue punctuation onto the previous word
                if words and all(not c.isalnum() for c in tok.text):
                    words[-1][2] += tok.text
                    continue
                if tok.start_ts is None or tok.end_ts is None:
                    continue
                words.append([round(t0 + tok.start_ts, 3), round(t0 + tok.end_ts, 3), tok.text])
            chunks += [audio, np.zeros(int(GAP * RATE), dtype=np.float32)]
            t0 += len(audio) / RATE + GAP
        raw = out / f"{seg['id']}_raw.wav"
        wav = out / f"{seg['id']}.wav"
        sf.write(raw, np.concatenate(chunks), RATE)
        p = float(cfg["pitch"])
        # raise pitch, keep speed: speed up by p via sample rate, then slow back down
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af",
                        f"asetrate={int(RATE * p)},aresample={RATE},atempo={1 / p:.5f},"
                        "highpass=f=90,acompressor=threshold=0.12:ratio=3:attack=5:release=60,volume=1.6",
                        str(wav)], check=True)
        raw.unlink()
        audio, _ = sf.read(wav, dtype="float32")
        hop = RATE // FPS
        env = [float(np.sqrt(np.mean(audio[i:i + hop] ** 2))) for i in range(0, len(audio), hop)]
        peak = max(env) or 1.0
        info[seg["id"]] = {"words": words, "env": [round(e / peak, 3) for e in env]}
        durations[seg["id"]] = round(len(audio) / RATE, 2)
        print(seg["id"], durations[seg["id"]], "s,", len(words), "words")

    Path("out/durations.json").write_text(json.dumps(durations, indent=1))
    Path("out/voice.json").write_text(json.dumps(info))
    Path("out/voice_lang.txt").write_text("en")


if __name__ == "__main__":
    main(sys.argv[1])
