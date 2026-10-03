"""Voices for a puppet show (Kokoro-82M, free, Apache-2.0, CPU).

    python pipeline/tts_show.py shows/ep01-three-jars

Speaks every beat's `say` line with the voice of its speaker (`who`), as set
in show.yaml under `voices:` ({voice, speed, pitch}). Writes:
  out/show/<beat id>.wav   one file per line (24 kHz mono)
  out/show/voice.json      per beat: seconds, word timings, loudness at 30 fps
The renderer uses the timings for captions and the loudness for mouths.
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
GAP = 0.22


def main(show_dir):
    show = yaml.safe_load((Path(show_dir) / "show.yaml").read_text(encoding="utf-8"))
    out = Path("out/show")
    out.mkdir(parents=True, exist_ok=True)
    pipe = KPipeline(lang_code="a")
    info = {}
    for beat in show["beats"]:
        text = beat.get("say")
        if not text:
            continue
        cfg = {"voice": "af_heart", "speed": 1.0, "pitch": 1.0, **(show.get("voices", {}).get(beat["who"]) or {})}
        chunks, words, t0 = [], [], 0.0
        for r in pipe(text, voice=cfg["voice"], speed=float(cfg["speed"]), split_pattern=r"(?<=[.!?])\s+"):
            audio = r.audio.detach().cpu().numpy() if hasattr(r.audio, "detach") else np.asarray(r.audio)
            # trim the model's own silence at both ends, so lines keep a brisk cartoon pace
            loud = np.where(np.abs(audio) > 0.02 * (np.abs(audio).max() + 1e-9))[0]
            a0 = max(0, int(loud[0]) - int(0.03 * RATE)) if len(loud) else 0
            a1 = min(len(audio), int(loud[-1]) + int(0.06 * RATE)) if len(loud) else len(audio)
            audio = audio[a0:a1]
            t0 -= a0 / RATE
            for tok in (r.tokens or []):
                if not tok.text.strip():
                    continue
                if words and all(not c.isalnum() for c in tok.text):
                    words[-1][2] += tok.text
                    continue
                if tok.start_ts is None or tok.end_ts is None:
                    continue
                words.append([round(max(0.0, t0 + tok.start_ts), 3), round(max(0.0, t0 + tok.end_ts), 3), tok.text])
            chunks += [audio, np.zeros(int(GAP * RATE), dtype=np.float32)]
            t0 += a0 / RATE + len(audio) / RATE + GAP
        raw = out / f"{beat['id']}_raw.wav"
        wav = out / f"{beat['id']}.wav"
        sf.write(raw, np.concatenate(chunks), RATE)
        p = float(cfg["pitch"])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af",
                        f"asetrate={int(RATE * p)},aresample={RATE},atempo={1 / p:.5f},"
                        "highpass=f=80,acompressor=threshold=0.12:ratio=3:attack=5:release=60,volume=1.5",
                        str(wav)], check=True)
        raw.unlink()
        audio, _ = sf.read(wav, dtype="float32")
        hop = RATE // FPS
        env = [float(np.sqrt(np.mean(audio[i:i + hop] ** 2))) for i in range(0, len(audio), hop)]
        peak = float(np.percentile(env, 95)) or 1.0
        info[beat["id"]] = {"dur": round(len(audio) / RATE, 3), "words": words,
                            "env": [round(min(1.0, e / peak), 3) for e in env]}
        print(beat["id"], info[beat["id"]]["dur"], "s")
    (out / "voice.json").write_text(json.dumps(info))


if __name__ == "__main__":
    main(sys.argv[1])
