"""Make a vertical 9:16 Short (Reels/Shorts) from an episode.

  python pipeline/make_short.py episodes/<ep> <en|hi|ta>

Layout (1080x1920): title band on top, the cartoon scene in the middle with a
slow zoom/pan, the spoken line below as a popping caption, and the education
disclaimer at the very bottom. Helpers borrowed from MoneyPrinterTurbo (MIT)
live in borrowed_mpt.py.

Inputs, all optional except script.yaml:
  episodes/<ep>/images/<segment id>.png|jpg|webp   one picture per scene
      (or set `image:` on a segment). Missing pictures get a colour card.
  Hindi voice: out/audio/<id>.wav + out/durations.json from tts_hindi.py
  Your own voice: episodes/<ep>/voice_<lang>.m4a|mp3|wav|aac|ogg|opus
      -> scenes and captions follow your recording (Whisper finds the timing).
  Music: assets/music/*.mp3|m4a (only tracks you are licensed to use), or set
      `music: <file name>` in script.yaml. Played quietly under the voice.

Output: out/final/<ep>_<lang>_short.mp4 and a matching .srt caption file.
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from borrowed_mpt import (align_to_script, cover_box, split_sentences,  # noqa: E402
                          spring_scale, whisper_cues, zoom_factor)

W, H, FPS = 1080, 1920, 30
TITLE_H = 300
SCENE_Y, SCENE_H = 300, 1080
CAP_Y, CAP_H = 1395, 420
FOOT_Y = 1835
FADE = 0.3            # cross-fade between scenes, seconds
LEAD = 0.3            # silence before each voice line, seconds (matches finish.py)
MUSIC_VOL = 0.12      # MPT default is 0.2; quieter so kids hear the words

BG = "#FFF3D6"
INK = "#1B1036"
BAND = [("#FF7A1A", "#FFB21F"), ("#4D96FF", "#6BCBFF"), ("#6BCB77", "#B5E655"), ("#FF6B9A", "#FFA6C9")]
CARD = ["#4D96FF", "#FF6B6B", "#6BCB77", "#FFB21F", "#9B6BFF"]
DISCLAIMER = {
    "en": "For education only. Not investment advice.",
    "hi": "केवल शिक्षा के लिए। यह निवेश सलाह नहीं है।",
    "ta": "கல்விக்காக மட்டும். இது முதலீட்டு ஆலோசனை அல்ல.",
}
FONT_FAMILY = {"en": ["Noto Sans", "Poppins", "DejaVu Sans"],
               "hi": ["Noto Sans Devanagari", "Poppins", "Lohit Devanagari"],
               "ta": ["Noto Sans Tamil", "Lohit Tamil"]}


# ---------------------------------------------------------------- helpers ---
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration_of(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def find_font(lang, bold=True):
    for fam in FONT_FAMILY[lang]:
        r = subprocess.run(["fc-match", "-f", "%{family}|%{file}",
                            f"{fam}:style={'Bold' if bold else 'Regular'}"],
                           capture_output=True, text=True)
        got_family, _, path = r.stdout.partition("|")
        if fam.lower() in got_family.lower() and path:
            return path
    raise SystemExit(f"No font for {lang}. Install fonts-noto-core.")


class Fonts:
    def __init__(self, lang):
        self.bold, self.reg, self.cache = find_font(lang, True), find_font(lang, False), {}

    def get(self, size, bold=True):
        key = (size, bold)
        if key not in self.cache:
            self.cache[key] = ImageFont.truetype(self.bold if bold else self.reg, size,
                                                 layout_engine=ImageFont.Layout.RAQM)
        return self.cache[key]


# Hindi videos: Hindi voice, but English (or Hinglish) words on screen.
# Add a "hinglish:" line next to en/hi in script.yaml to use Hinglish instead.
SCREEN_TEXT = {"hi": ["hinglish", "en"]}


def pick(field, lang):
    if isinstance(field, dict):
        for k in SCREEN_TEXT.get(lang, []):
            if field.get(k):
                return field[k]
        return field.get(lang) or field.get("en") or ""
    return field or ""


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        if cur and draw.textlength(test, font=font) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines


def fit_text(draw, text, fonts, max_w, max_h, start=72, low=34, max_lines=3, bold=True):
    """Biggest font size where the text fits the box."""
    size = start
    while size >= low:
        f = fonts.get(size, bold)
        lines = wrap(draw, text, f, max_w)
        line_h = int(size * 1.35)
        if len(lines) <= max_lines and line_h * len(lines) <= max_h:
            return f, lines, line_h
        size -= 4
    f = fonts.get(low, bold)
    return f, wrap(draw, text, f, max_w), int(low * 1.35)


def gradient(w, h, top, bottom):
    a, b = ImageColor.getrgb(top), ImageColor.getrgb(bottom)
    col = Image.new("RGB", (1, h))
    for y in range(h):
        k = y / max(h - 1, 1)
        col.putpixel((0, y), tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3)))
    return col.resize((w, h))


def caption_lines(text):
    """Caption-sized chunks: split at full stops, then at commas if still long."""
    out = []
    for s in split_sentences(text, keep_commas=True):
        out += split_sentences(s) if len(s) > 70 else [s]
    return out or [text]


# ---------------------------------------------------------------- timing ----
def find_voice(ep, lang):
    for ext in ("m4a", "mp3", "wav", "aac", "ogg", "opus", "webm"):
        p = ep / f"voice_{lang}.{ext}"
        if p.exists():
            return p
    return None


def plan_timeline(script, ep, lang):
    """Return (segments, audio_plan). Each segment: dict(seg, start, dur, cues)."""
    segs = script["segments"]
    durations = {}
    dpath = Path("out/durations.json")
    if lang == "hi" and dpath.exists():
        durations = json.loads(dpath.read_text())
    voice = find_voice(ep, lang)

    if voice and lang != "hi":
        return plan_from_recording(segs, voice, lang)

    t, plan, clips = 0.0, [], []
    for seg in segs:
        spoken = durations.get(seg["id"], 0)
        total = max(float(seg.get("seconds", 5)), spoken + 2 * LEAD if spoken else 0)
        lines = caption_lines(pick(seg.get("say"), lang))
        talk = spoken if spoken else total - 2 * LEAD
        plan.append(dict(seg=seg, start=t, dur=total,
                         cues=spread(lines, t + LEAD, talk)))
        wav = Path(f"out/audio/{seg['id']}.wav")
        if spoken and wav.exists():
            clips.append((wav, t + LEAD))
        t += total
    return plan, dict(kind="clips" if clips else "none", clips=clips, total=t)


def spread(lines, start, length):
    """Share the time between lines by how long each line is (MPT's fallback idea)."""
    total_chars = sum(len(x) for x in lines) or 1
    cues, t = [], start
    for x in lines:
        d = length * len(x) / total_chars
        cues.append((t, t + d, x))
        t += d
    return cues


def plan_from_recording(segs, voice, lang):
    total = duration_of(voice)
    lines = []                          # (segment index, text)
    for i, seg in enumerate(segs):
        lines += [(i, x) for x in caption_lines(pick(seg.get("say"), lang))]
    texts = [x for _, x in lines]
    cues = whisper_cues(str(voice), lang)
    times = align_to_script(cues, texts) if cues else [None] * len(texts)
    if not cues:
        print("Whisper not available: sharing time by line length.")
    # fill gaps and keep times in order
    fallback = spread(texts, LEAD, total - LEAD)
    fixed, last = [], 0.0
    for k, tm in enumerate(times):
        s, e = tm if tm else fallback[k][:2]
        s = max(s, last)
        e = max(e, s + 0.4)
        fixed.append((s, e))
        last = s
    # captions stay up until the next line starts
    cue_list = []
    for k, (s, e) in enumerate(fixed):
        nxt = fixed[k + 1][0] if k + 1 < len(fixed) else total + 0.5
        cue_list.append((s, max(e, nxt), texts[k], lines[k][0]))
    plan = []
    for i, seg in enumerate(segs):
        mine = [c for c in cue_list if c[3] == i]
        start = 0.0 if i == 0 else (mine[0][0] - 0.15 if mine else plan[-1]["start"] + plan[-1]["dur"])
        plan.append(dict(seg=seg, start=start, dur=0, cues=[c[:3] for c in mine]))
    for i, p in enumerate(plan):
        end = plan[i + 1]["start"] if i + 1 < len(plan) else total + 0.6
        p["dur"] = max(end - p["start"], 0.5)
    return plan, dict(kind="recording", file=voice, total=total + 0.6)


# ---------------------------------------------------------------- pictures --
def find_image(ep, seg):
    if seg.get("image"):
        p = Path(seg["image"])
        return p if p.is_absolute() or p.exists() else ep / p
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = ep / "images" / f"{seg['id']}.{ext}"
        if p.exists():
            return p
    return None


def placeholder(seg, idx, lang, fonts):
    size = int(SCENE_H * HEADROOM)          # same size as the zoom window, so text never crops
    img = gradient(size, size, CARD[idx % len(CARD)], "#1B1036")
    d = ImageDraw.Draw(img)
    text = pick(seg.get("heading"), lang) or pick(seg.get("say"), lang)
    f, lines, lh = fit_text(d, text, fonts, SCENE_H * 0.8, SCENE_H * 0.6, start=84, max_lines=5)
    y = (size - lh * len(lines)) // 2
    for line in lines:
        d.text((size // 2, y), line, font=f, fill="white", anchor="ma",
               stroke_width=5, stroke_fill="black")
        y += lh
    d.text((size // 2, size - (size - SCENE_H) // 2 - 110), f"add picture: images/{seg['id']}.png",
           font=fonts.get(30, False), fill="#FFFFFF", anchor="ma")
    return img


HEADROOM = 1.16     # extra picture around the frame so we can zoom and pan


def prepare_scene(path, seg, idx, lang, fonts):
    src = Image.open(path).convert("RGB") if path else placeholder(seg, idx, lang, fonts)
    tw, th = int(SCENE_H * HEADROOM), int(SCENE_H * HEADROOM)   # square scene window
    scale, cx, cy = cover_box(src.width, src.height, tw, th)
    big = src.resize((max(tw, round(src.width * scale)), max(th, round(src.height * scale))),
                     Image.LANCZOS)
    return big.crop((cx, cy, cx + tw, cy + th))


# Moves rotate scene by scene so episodes don't feel samey.
MOVES = ["zoom_in", "pan_right", "zoom_out", "pan_left"]


def scene_frame(big, t, dur, move):
    z = zoom_factor(t, dur)                      # 1.0 -> about 1.15
    p = t / max(dur, 1e-6)
    if move == "zoom_out":
        z = zoom_factor(dur - t, dur)
    base = big.width / HEADROOM                  # the window at z = 1
    win = base / (z if move.startswith("zoom") else 1.0)
    spare = big.width - win
    cx = cy = spare / 2
    if move == "pan_right":
        cx = spare * p
    elif move == "pan_left":
        cx = spare * (1 - p)
    return big.resize((SCENE_H, SCENE_H), Image.BILINEAR, box=(cx, cy, cx + win, cy + win))


# ---------------------------------------------------------------- layers ----
def make_base(title, lang, fonts, band):
    base = Image.new("RGB", (W, H), BG)
    base.paste(gradient(W, TITLE_H, *band), (0, 0))
    d = ImageDraw.Draw(base)
    f, lines, lh = fit_text(d, title, fonts, W - 120, TITLE_H - 60, start=86, max_lines=2)
    y = (TITLE_H - lh * len(lines)) // 2 + 6
    for line in lines:
        d.text((W // 2, y), line, font=f, fill="white", anchor="ma", stroke_width=7, stroke_fill=INK)
        y += lh
    d.rectangle((0, SCENE_Y - 8, W, SCENE_Y + SCENE_H + 8), fill=INK)     # thick cartoon frame
    d.text((W // 2, FOOT_Y + 28), DISCLAIMER[lang], font=fonts.get(30, False), fill="#6B5E80", anchor="ma")
    return base


def make_caption(text, fonts):
    layer = Image.new("RGBA", (W, CAP_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f, lines, lh = fit_text(d, text, fonts, W - 170, CAP_H - 90, start=68, max_lines=3)
    box_h = lh * len(lines) + 70
    box_w = min(W - 60, int(max(d.textlength(x, font=f) for x in lines)) + 110)
    x0, y0 = (W - box_w) // 2, (CAP_H - box_h) // 2
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((x0 + 8, y0 + 10, x0 + box_w + 8, y0 + box_h + 10),
                                             radius=40, fill=(27, 16, 54, 110))
    layer = Image.alpha_composite(layer, shadow.filter(ImageFilter.GaussianBlur(6)))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=40, fill="white", outline=INK, width=7)
    y = y0 + 35
    for line in lines:
        d.text((W // 2, y), line, font=f, fill=INK, anchor="ma")
        y += lh
    return layer


def scaled(layer, s):
    if s == 1.0:
        return layer
    w, h = max(1, int(layer.width * s)), max(1, int(layer.height * s))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.paste(layer.resize((w, h), Image.BILINEAR), ((layer.width - w) // 2, (layer.height - h) // 2))
    return out


# ---------------------------------------------------------------- audio -----
def build_audio(audio, music, total, dest):
    cmd, filt, n = ["ffmpeg", "-y"], [], 0
    voice_label = None
    if audio["kind"] == "clips":
        labels = []
        for wav, at in audio["clips"]:
            cmd += ["-i", str(wav)]
            ms = int(at * 1000)
            filt.append(f"[{n}:a]aformat=sample_rates=44100:channel_layouts=stereo,adelay={ms}:all=1[v{n}]")
            labels.append(f"[v{n}]")
            n += 1
        filt.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest[voice]")
        voice_label = "[voice]"
    elif audio["kind"] == "recording":
        cmd += ["-i", str(audio["file"])]
        filt.append(f"[{n}:a]aformat=sample_rates=44100:channel_layouts=stereo[voice]")
        voice_label, n = "[voice]", n + 1
    if music and voice_label:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        filt.append(f"[{n}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={MUSIC_VOL},"
                    f"atrim=0:{total:.2f},afade=t=out:st={max(total - 2.5, 0):.2f}:d=2.5[mus]")
        filt.append(f"{voice_label}[mus]amix=inputs=2:normalize=0:duration=first[mix]")
        voice_label, n = "[mix]", n + 1
    if not voice_label:      # silent track so Reels/Shorts accept the file
        cmd += ["-f", "lavfi", "-t", f"{total:.2f}", "-i", "anullsrc=r=44100:cl=stereo",
                "-c:a", "aac", str(dest)]
        run(cmd)
        return
    # YouTube plays at about -14 LUFS; level the voice to that.
    filt.append(f"{voice_label}apad,atrim=0:{total:.2f},loudnorm=I=-14:TP=-1.5:LRA=11[out]")
    cmd += ["-filter_complex", ";".join(filt), "-map", "[out]", "-ar", "44100", "-c:a", "aac",
            "-b:a", "192k", str(dest)]
    run(cmd)


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------- main ------
def main(ep_dir, lang):
    ep = Path(ep_dir)
    script = yaml.safe_load((ep / "script.yaml").read_text(encoding="utf-8"))
    text_lang = "en" if lang in SCREEN_TEXT else lang
    fonts = Fonts(text_lang)
    segs = script["segments"]
    title = (pick(script.get("short_title"), lang)
             or pick(next((s.get("heading") for s in segs if s.get("visual") == "title"), None), lang)
             or pick(script.get("title"), lang))

    plan, audio = plan_timeline(script, ep, lang)
    total = audio["total"]
    band = BAND[sum(map(ord, ep.name)) % len(BAND)]
    base = make_base(title, text_lang, fonts, band)
    scenes = [prepare_scene(find_image(ep, p["seg"]), p["seg"], i, lang, fonts) for i, p in enumerate(plan)]
    cues = [(s, e, x) for p in plan for (s, e, x) in p["cues"]]
    cap_layers = [make_caption(x, fonts) for _, _, x in cues]

    music = None
    mdir = Path("assets/music")
    if script.get("music"):
        music = mdir / script["music"]
    elif mdir.exists():
        music = next(iter(sorted(p for p in mdir.iterdir() if p.suffix.lower() in (".mp3", ".m4a", ".wav"))), None)
    if audio["kind"] == "none":
        music = None          # silent version is for recording your voice over

    work = Path("out/short"); work.mkdir(parents=True, exist_ok=True)
    final = Path("out/final"); final.mkdir(parents=True, exist_ok=True)
    audio_file = work / f"{lang}_audio.m4a"
    build_audio(audio, music, total, audio_file)

    target = final / f"{ep.name}_{lang}_short.mp4"
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-i", str(audio_file), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-movflags", "+faststart", "-shortest", str(target)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = int(total * FPS)
    si, ci = 0, 0
    for k in range(frames):
        t = k / FPS
        while si + 1 < len(plan) and t >= plan[si + 1]["start"]:
            si += 1
        p = plan[si]
        local = t - p["start"]
        img = scene_frame(scenes[si], local, p["dur"], MOVES[si % len(MOVES)])
        if si > 0 and local < FADE:
            q = plan[si - 1]
            prev = scene_frame(scenes[si - 1], q["dur"], q["dur"], MOVES[(si - 1) % len(MOVES)])
            img = Image.blend(prev, img, local / FADE)
        frame = base.copy()
        frame.paste(img, (0, SCENE_Y))
        while ci + 1 < len(cues) and t >= cues[ci + 1][0]:
            ci += 1
        if cues and cues[ci][0] <= t < cues[ci][1]:
            layer = scaled(cap_layers[ci], spring_scale(t - cues[ci][0]))
            frame.paste(layer, (0, CAP_Y), layer)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg failed while encoding the Short")

    srt = [f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{x}\n" for i, (s, e, x) in enumerate(cues, 1)]
    target.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")
    print(f"wrote {target} ({total:.1f}s, {len(plan)} scenes, {len(cues)} captions, "
          f"voice={audio['kind']}, music={'yes' if music else 'no'})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
