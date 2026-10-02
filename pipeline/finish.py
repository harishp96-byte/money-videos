"""Final assembly after rendering.

  hi: lay each segment's Hindi voice at the start of its scene, add it to the
      video -> out/final/<ep>_hi.mp4
  en/ta: keep the video silent for your own voiceover and write a
      read-along file with the exact start time of every line
      -> out/final/<ep>_<lang>.mp4 and <ep>_<lang>_readalong.txt
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def seg_times(script, durations):
    t, out = 0.0, []
    for seg in script["segments"]:
        spoken = durations.get(seg["id"], 0)
        total = max(float(seg.get("seconds", 5)), spoken + 0.6 if spoken else 0)
        out.append((seg, t, total))
        t += total
    return out


def main(ep_dir, lang, video):
    ep = Path(ep_dir)
    name = ep.name
    script = yaml.safe_load((ep / "script.yaml").read_text(encoding="utf-8"))
    dur_file = Path("out/durations.json")
    durations = json.loads(dur_file.read_text()) if dur_file.exists() else {}
    final = Path("out/final")
    final.mkdir(parents=True, exist_ok=True)
    target = final / f"{name}_{lang}.mp4"
    times = seg_times(script, durations)

    if lang == "hi" and durations:
        rate = None
        length = sum(total for _, _, total in times)
        track = None
        for seg, start, _ in times:
            wav = Path(f"out/audio/{seg['id']}.wav")
            if not wav.exists():
                continue
            audio, r = sf.read(wav, dtype="float32")
            if track is None:
                rate = r
                track = np.zeros(int((length + 1) * rate), dtype=np.float32)
            i = int((start + 0.3) * rate)
            track[i:i + len(audio)] += audio[: len(track) - i]
        sf.write("out/hi_voice.wav", track, rate)
        subprocess.run(["ffmpeg", "-y", "-i", video, "-i", "out/hi_voice.wav", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", str(target)], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-i", video, "-c", "copy", str(target)], check=True)
        lines = [f"{script.get('title', name)} - read-along ({lang})", ""]
        for seg, start, total in times:
            m, s = divmod(int(start), 60)
            lines.append(f"[{m}:{s:02d}] ({total:.0f}s)  {seg.get('say', {}).get(lang, '')}")
        (final / f"{name}_{lang}_readalong.txt").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", target)


if __name__ == "__main__":
    main(*sys.argv[1:4])
