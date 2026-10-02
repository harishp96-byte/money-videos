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
    """video: the 16:9 render. A matching <lang>_short.mp4 next to it is
    finished the same way (cartoon renderer makes both)."""
    short = Path(video).with_name(Path(video).stem + "_short.mp4")
    finish_one(ep_dir, lang, video, "")
    if short.exists():
        finish_one(ep_dir, lang, str(short), "_short", readalong=False)


def has_audio(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index", "-of", "csv=p=0", video], capture_output=True, text=True)
    return bool(r.stdout.strip())


def finish_one(ep_dir, lang, video, suffix, readalong=True):
    ep = Path(ep_dir)
    name = ep.name
    script = yaml.safe_load((ep / "script.yaml").read_text(encoding="utf-8"))
    dur_file = Path("out/durations.json")
    durations = json.loads(dur_file.read_text()) if dur_file.exists() else {}
    final = Path("out/final")
    final.mkdir(parents=True, exist_ok=True)
    target = final / f"{name}_{lang}{suffix}.mp4"
    times = seg_times(script, durations)

    vl = Path("out/voice_lang.txt")
    voiced = vl.read_text().strip() if vl.exists() else "hi"
    if lang == voiced and durations:
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
        if has_audio(video):   # keep the cartoon sound effects under the voice
            subprocess.run(["ffmpeg", "-y", "-i", video, "-i", "out/hi_voice.wav", "-filter_complex",
                            "[0:a]volume=0.6[a0];[a0][1:a]amix=inputs=2:duration=first:normalize=0[a]",
                            "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            str(target)], check=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-i", video, "-i", "out/hi_voice.wav", "-c:v", "copy",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", str(target)], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-i", video, "-c", "copy", str(target)], check=True)
        if not readalong:
            print("wrote", target)
            return
        lines = [f"{script.get('title', name)} - read-along ({lang})", ""]
        for seg, start, total in times:
            m, s = divmod(int(start), 60)
            lines.append(f"[{m}:{s:02d}] ({total:.0f}s)  {seg.get('say', {}).get(lang, '')}")
        (final / f"{name}_{lang}_readalong.txt").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", target)


if __name__ == "__main__":
    main(*sys.argv[1:4])
