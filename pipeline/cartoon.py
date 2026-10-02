"""Cartoon renderer (v2): turns episodes/<ep>/script.yaml into a kids'-cartoon
style video in two native shapes:

  out/media/cartoon/<lang>.mp4        16:9 long video (1920x1080)
  out/media/cartoon/<lang>_short.mp4  9:16 Short/Reel (1080x1920), full frame

What makes it look like a show rather than a slideshow (v2 changes):
  * Every shot is composed separately for each shape: the Short is a real
    full-frame vertical scene, not a 16:9 panel inside black bars.
  * Rooms are built in layers (back wall/furniture, actors, foreground leaves)
    and the camera moves each layer at a different speed (2.5D parallax).
  * The boy stands ON the floor: soft contact shadow, a cast shadow away from
    the window light, warm room tint, rim light on the window side, and his
    size follows the floor perspective.
  * He keeps moving: breathing, blinking, leaning from the feet, step-bob when
    walking, squash on landing, mouth flaps while talking.
  * Ideas are acted out with props that sit on tables and floors (salary
    envelope -> notes fly into three jars; a stack of notes -> piggy bank with
    the counter rising out of it; piggy / phone / shopping bag for tips).
  * Camera beats: each shot has its own push-in or drift; punch-in on cuts.
  * Captions are word-by-word "karaoke" (active word in yellow), kept inside
    the YouTube Shorts safe area, and move above the face in close-ups.

Usage:  python pipeline/cartoon.py <episode_dir> <lang> [--preview SECONDS] [--only short|long]
Env:    DURATIONS=out/durations.json (voice lengths, optional)
"""
import glob
import json
import math
import os
import random
import subprocess
import sys
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FPS = 30
SPRITES = Path("assets/characters/sprites")

INK = (43, 27, 23)
GOLD, GOLD_D = (255, 196, 46), (214, 138, 0)
NEED, WANT, SAVE, POP = (255, 107, 107), (255, 211, 61), (88, 204, 102), (77, 150, 255)
YELLOW_T = (255, 214, 10)
SKIN = (238, 160, 106)
SHADOW = (70, 40, 25)

# ---------------------------------------------------------------- script ---
EP = Path(sys.argv[1] if len(sys.argv) > 1 else "episodes/week00-test")
LANG = sys.argv[2] if len(sys.argv) > 2 else "ta"
PREVIEW = float(sys.argv[sys.argv.index("--preview") + 1]) if "--preview" in sys.argv else None
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
SCRIPT = yaml.safe_load((EP / "script.yaml").read_text(encoding="utf-8"))
DUR = {}
if os.environ.get("DURATIONS") and Path(os.environ["DURATIONS"]).exists():
    DUR = json.loads(Path(os.environ["DURATIONS"]).read_text())
TEXT_LANG = {"hi": "en"}.get(LANG, LANG)   # Hindi: Hindi voice, English text

# Voiceover data from tts_en.py: word timings + loudness per segment, used for
# caption timing and lip movement. Only applies to the language that was voiced.
VOICE = {}
_vl = Path("out/voice_lang.txt")
if Path("out/voice.json").exists() and _vl.exists() and _vl.read_text().strip() == LANG:
    VOICE = json.loads(Path("out/voice.json").read_text())
VOICE_LEAD = 0.3          # finish.py starts each line 0.3 s into its scene
SEG_START = {}            # segment id -> start time (filled in main)
NOW = [0.0]               # current video time (lip sync, blinks)


def t(field):
    if not isinstance(field, dict):
        return field
    if LANG == "hi" and field.get("hinglish"):
        return field["hinglish"]
    return field.get(TEXT_LANG) or field.get("en")


def caps(s):
    return s.upper() if TEXT_LANG == "en" else s


def seg_time(seg):
    spoken = DUR.get(seg["id"], 0)
    return max(float(seg.get("seconds", 5)), spoken + 0.6 if spoken else 0)


# ----------------------------------------------------------------- stage ---
class Stage:
    """One output shape. Shots read sizes and layout points from here."""

    def __init__(self, kind):
        self.kind = kind
        self.port = kind == "short"
        self.w, self.h = (1080, 1920) if self.port else (1920, 1080)
        self.hz = 980 if self.port else 720          # where the floor meets the wall
        self.base = 760 if self.port else 640         # boy height scale

    def P(self, port, land):
        return port if self.port else land

    def depth(self, y):
        return clamp((y - self.hz) / (self.h - self.hz))

    def boy_h(self, foot_y):
        """Full-body height in px for feet at foot_y (nearer = bigger)."""
        return self.base * (0.62 + 0.55 * self.depth(foot_y))


SHORT, LONG = Stage("short"), Stage("long")


# ----------------------------------------------------------------- fonts ---
def find_font(names):
    files = glob.glob("/usr/share/fonts/**/*.[ot]tf", recursive=True)
    files += glob.glob(os.path.expanduser("~/.fonts/**/*.[ot]tf"), recursive=True)
    by_name = {Path(f).name.lower(): f for f in files}
    for n in names:
        if n.lower() in by_name:
            return by_name[n.lower()]
    raise SystemExit(f"No font found from {names}. Install fonts-noto-core.")


LATIN = find_font(["NotoSans-CondensedBlack.ttf", "NotoSans-CondensedExtraBold.ttf",
                   "NotoSans-Black.ttf", "NotoSans-Bold.ttf", "Inter-Black.otf",
                   "DejaVuSansCondensed-Bold.ttf", "DejaVuSans-Bold.ttf"])
LATIN_SYM = find_font(["NotoSans-Black.ttf", "NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf"])  # has ₹
TAMIL = find_font(["NotoSansTamil-Black.ttf", "NotoSansTamil-ExtraBold.ttf",
                   "NotoSansTamil-Bold.ttf", "FreeSans.ttf"])
FAKE_BOLD = Path(TAMIL).name == "FreeSans.ttf"      # local preview only


@lru_cache(maxsize=256)
def font(path, size):
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)


def runs(text):
    """Split text into (font_path, chunk) runs: Tamil script vs everything else."""
    out = []
    for ch in text:
        if "஀" <= ch <= "௿":
            f = TAMIL
        elif ch in "₹✓":
            f = LATIN_SYM
        elif ch in " ,.!?-:;'\"%()" and out:
            f = out[-1][0]
        else:
            f = LATIN
        if out and out[-1][0] == f:
            out[-1][1] += ch
        else:
            out.append([f, ch])
    return out


def text_w(text, size):
    return sum(font(f, size).getlength(c) for f, c in runs(text))


@lru_cache(maxsize=2048)
def text_img(text, size, fill=(255, 255, 255), stroke=0, stroke_fill=INK, shadow=0):
    """One line of text as a tight RGBA sprite."""
    pad = stroke + shadow + 8
    w = int(text_w(text, size)) + pad * 2
    h = int(size * 1.6) + pad * 2
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    base = pad + int(size * 1.15)
    for dx, dy, col, sw in ([(shadow, shadow, stroke_fill, stroke)] if shadow else []) + [(0, 0, fill, stroke)]:
        x = pad
        for f, c in runs(text):
            fb = 2 if (FAKE_BOLD and f == TAMIL) else 0
            d.text((x + dx, base + dy), c, font=font(f, size), fill=col, anchor="ls",
                   stroke_width=sw + fb, stroke_fill=stroke_fill if sw else col)
            x += font(f, size).getlength(c)
    return im.crop(im.getbbox())


def wrap(text, size, max_w):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        test = f"{cur} {w_}".strip()
        if cur and text_w(test, size) > max_w:
            lines.append(cur)
            cur = w_
        else:
            cur = test
    lines.append(cur)
    return lines


def text_block(text, max_w, start, low, max_lines=3, **kw):
    size = start
    while size > low:
        lines = wrap(text, size, max_w)
        if len(lines) <= max_lines and all(text_w(l, size) <= max_w for l in lines):
            break
        size -= 4
    lines = wrap(text, size, max_w)
    ims = [text_img(l, size, **kw) for l in lines]
    gap = int(size * 0.18)
    bw, bh = max(i.width for i in ims), sum(i.height for i in ims) + gap * (len(ims) - 1)
    out = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    y = 0
    for i in ims:
        out.alpha_composite(i, ((bw - i.width) // 2, y))
        y += i.height + gap
    return fit(out, max_w)


def fit(im, max_w, max_h=None):
    s = min(1.0, max_w / im.width, (max_h / im.height) if max_h else 1.0)
    if s < 0.999:
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return im


# ---------------------------------------------------------------- easing ---
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x)
    return 3 * x * x - 2 * x * x * x


def back(x, s=2.2):
    x = clamp(x)
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


def pop(local, start, dur=0.35):
    """Scale for a pop-in that overshoots then settles."""
    if local < start:
        return 0.0
    return back((local - start) / dur)


def lerp(a, b, x):
    return a + (b - a) * x


def keys(local, ks):
    """Camera keyframes [(time, zoom, fx, fy), ...] -> eased (zoom, fx, fy)."""
    if local <= ks[0][0]:
        return ks[0][1:]
    for (t0, *a), (t1, *b) in zip(ks, ks[1:]):
        if local <= t1:
            x = ease_io((local - t0) / max(1e-6, t1 - t0))
            return tuple(lerp(p, q, x) for p, q in zip(a, b))
    return ks[-1][1:]


# --------------------------------------------------------------- drawing ---
def hand_drawn(w, h, fn, ss=2):
    """Draw at 2x with fn(draw, scale) then shrink: smooth bold outlines."""
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    fn(ImageDraw.Draw(im), ss)
    return im.resize((w, h), Image.LANCZOS) if ss != 1 else im


def place(frame, sprite, cx, cy, scale=1.0, rot=0.0, anchor="center", alpha=1.0):
    """Paste sprite with its center (or bottom-center) at cx, cy."""
    if scale <= 0.01 or sprite is None:
        return
    sp = sprite
    if abs(scale - 1) > 0.01:
        sp = sp.resize((max(1, int(sp.width * scale)), max(1, int(sp.height * scale))), Image.BILINEAR)
    if abs(rot) > 0.3:
        sp = sp.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 0.99:
        sp = sp.copy()
        sp.putalpha(sp.getchannel("A").point(lambda v: int(v * alpha)))
    x = int(cx - sp.width / 2)
    y = int(cy - sp.height / 2) if anchor == "center" else int(cy - sp.height)
    if x >= frame.width or y >= frame.height or x + sp.width <= 0 or y + sp.height <= 0:
        return                                   # completely off screen
    if frame.mode == "RGBA":
        frame.alpha_composite(sp, (max(0, x), max(0, y)), (max(0, -x), max(0, -y)))
    else:
        frame.paste(sp, (x, y), sp)


@lru_cache(maxsize=256)
def soft_ellipse(w, h, alpha, blur):
    """Blurred dark ellipse used for every contact shadow."""
    pad = blur * 3 + 2
    im = Image.new("L", (w + pad * 2, h + pad * 2), 0)
    ImageDraw.Draw(im).ellipse([pad, pad, pad + w, pad + h], fill=int(255 * alpha))
    im = im.filter(ImageFilter.GaussianBlur(blur))
    out = Image.new("RGBA", im.size, SHADOW + (0,))
    out.putalpha(im)
    return out


def ground(layer, x, y, w, alpha=0.38, flat=0.16):
    """Contact shadow: soft ellipse centred under an object standing at (x, y)."""
    w = max(8, int(w))
    sh = soft_ellipse(w, max(4, int(w * flat)), round(alpha, 2), max(3, w // 22))
    place(layer, sh, x + w * 0.04, y)


_DROP = {}


def drop_cached(key, sprite, *a):
    if key not in _DROP:
        _DROP[key] = with_drop(sprite, *a)
    return _DROP[key]


def with_drop(sprite, dx=10, dy=12, alpha=0.30, blur=6):
    """Sprite plus a soft drop shadow down-right (light comes from the window, top-left)."""
    pad = blur * 3 + max(dx, dy)
    out = Image.new("RGBA", (sprite.width + pad * 2, sprite.height + pad * 2), (0, 0, 0, 0))
    a = sprite.getchannel("A").point(lambda v: int(v * alpha))
    sh = Image.new("RGBA", sprite.size, SHADOW + (0,))
    sh.putalpha(a)
    out.alpha_composite(sh, (pad + dx, pad + dy))
    out = out.filter(ImageFilter.GaussianBlur(blur)) if blur else out
    out.alpha_composite(sprite, (pad, pad))
    return out


@lru_cache(maxsize=None)
def coin_sprite(r=46):
    def f(d, s):
        R = r * s
        d.ellipse([4 * s, 4 * s, 2 * R - 4 * s, 2 * R - 4 * s], fill=GOLD, outline=INK, width=5 * s)
        d.ellipse([R * 0.32, R * 0.32, R * 1.68, R * 1.68], outline=GOLD_D, width=4 * s)
        d.text((R, R * 1.06), "₹", font=font(LATIN_SYM, int(R * 1.05)), fill=(122, 75, 0), anchor="mm")
    return hand_drawn(2 * r, 2 * r, f)


@lru_cache(maxsize=None)
def note_sprite():
    w, h = 220, 110

    def f(d, s):
        d.rounded_rectangle([4 * s, 4 * s, (w - 4) * s, (h - 4) * s], 12 * s, fill=(124, 214, 140), outline=INK, width=5 * s)
        d.rounded_rectangle([18 * s, 16 * s, (w - 18) * s, (h - 16) * s], 8 * s, outline=(50, 140, 70), width=3 * s)
        d.ellipse([(w / 2 - 30) * s, (h / 2 - 30) * s, (w / 2 + 30) * s, (h / 2 + 30) * s], fill=(205, 244, 210), outline=(50, 140, 70), width=3 * s)
        d.text((w / 2 * s, h / 2 * s + 3 * s), "₹", font=font(LATIN_SYM, 44 * s), fill=(40, 110, 55), anchor="mm")
    return hand_drawn(w, h, f)


@lru_cache(maxsize=8)
def note_stack(n=8):
    """A neat bundle of notes seen slightly from above, with a paper band."""
    w, step = 240, 12
    h = 90 + step * (n - 1)

    def f(d, s):
        for k in range(n):
            y = h - 90 - k * step + (n - 1) * step - (n - 1) * step
            y = (n - 1 - k) * step
            d.rounded_rectangle([6 * s, (y + 4) * s, (w - 6) * s, (y + 86) * s], 12 * s,
                                fill=(124, 214, 140) if k % 2 == 0 else (110, 200, 128), outline=INK, width=5 * s)
        d.rectangle([(w / 2 - 22) * s, 2 * s, (w / 2 + 22) * s, (h - 2) * s], fill=(250, 240, 210), outline=INK, width=4 * s)
        d.text((w / 2 * s, 46 * s), "₹", font=font(LATIN_SYM, 40 * s), fill=(40, 110, 55), anchor="mm")
    return hand_drawn(w, h, f)


@lru_cache(maxsize=None)
def envelope_sprite():
    """Salary envelope lying on the table, flap open."""
    w, h = 300, 190

    def f(d, s):
        S = lambda *v: [x * s for x in v]
        d.polygon(S(20, 70, 280, 70, 230, 14, 70, 14), fill=(232, 200, 150), outline=INK, width=6 * s)     # open flap
        d.rounded_rectangle(S(8, 60, 292, 182), 14 * s, fill=(250, 222, 170), outline=INK, width=6 * s)
        d.line(S(12, 66, 150, 140, 288, 66), fill=(214, 170, 110), width=5 * s)
        d.ellipse(S(122, 112, 178, 168), fill=NEED, outline=INK, width=5 * s)
        d.text((150 * s, 141 * s), "₹", font=font(LATIN_SYM, 34 * s), fill=(255, 255, 255), anchor="mm")
    return hand_drawn(w, h, f)


@lru_cache(maxsize=None)
def calendar_sprite(num, word):
    w, h = 250, 280

    def f(d, s):
        d.rounded_rectangle([6 * s, 20 * s, (w - 6) * s, (h - 6) * s], 18 * s, fill=(255, 255, 255), outline=INK, width=6 * s)
        d.rounded_rectangle([6 * s, 20 * s, (w - 6) * s, 90 * s], 18 * s, fill=NEED, outline=INK, width=6 * s)
        d.rectangle([12 * s, 70 * s, (w - 12) * s, 88 * s], fill=NEED)
        for x in (70, w - 70):
            d.rounded_rectangle([(x - 9) * s, 4 * s, (x + 9) * s, 44 * s], 8 * s, fill=(90, 90, 100), outline=INK, width=4 * s)
        d.text((w / 2 * s, 180 * s), str(num), font=font(LATIN, 130 * s), fill=INK, anchor="mm")
    im = hand_drawn(w, h, f)
    lab = fit(text_img(word, 30, fill=(255, 255, 255)), w - 40)
    im.alpha_composite(lab, ((w - lab.width) // 2, 55 - lab.height // 2))
    return im


@lru_cache(maxsize=None)
def jar_sprite():
    w, h = 270, 360

    def f(d, s):
        d.rounded_rectangle([14 * s, 70 * s, (w - 14) * s, (h - 6) * s], 46 * s, fill=(225, 245, 255, 120), outline=INK, width=7 * s)
        d.rounded_rectangle([40 * s, 30 * s, (w - 40) * s, 80 * s], 14 * s, fill=(255, 140, 60), outline=INK, width=7 * s)
    return hand_drawn(w, h, f)


@lru_cache(maxsize=None)
def jar_glint():
    w, h = 270, 360

    def f(d, s):
        d.line([(50 * s, 120 * s), (50 * s, 260 * s)], fill=(255, 255, 255, 210), width=14 * s)
        d.line([(50 * s, 290 * s), (50 * s, 310 * s)], fill=(255, 255, 255, 210), width=14 * s)
    return hand_drawn(w, h, f)


@lru_cache(maxsize=None)
def piggy_sprite():
    w, h = 520, 400

    def f(d, s):
        P, PD = (255, 158, 196), (226, 95, 150)
        for x in (130, 360):  # legs
            d.rounded_rectangle([x * s, 290 * s, (x + 60) * s, 385 * s], 20 * s, fill=PD, outline=INK, width=7 * s)
        d.ellipse([40 * s, 70 * s, 480 * s, 340 * s], fill=P, outline=INK, width=8 * s)
        d.ellipse([90 * s, 110 * s, 220 * s, 170 * s], fill=(255, 200, 225))          # shine
        d.polygon([(130 * s, 95 * s), (170 * s, 20 * s), (215 * s, 85 * s)], fill=PD, outline=INK, width=7 * s)
        d.ellipse([400 * s, 160 * s, 505 * s, 255 * s], fill=PD, outline=INK, width=7 * s)
        for x in (432, 468):
            d.ellipse([(x - 9) * s, 196 * s, (x + 9) * s, 220 * s], fill=INK)
        d.ellipse([340 * s, 140 * s, 372 * s, 172 * s], fill=INK)
        d.ellipse([350 * s, 146 * s, 360 * s, 156 * s], fill=(255, 255, 255))
        d.rounded_rectangle([200 * s, 80 * s, 330 * s, 98 * s], 9 * s, fill=INK)
        d.arc([10 * s, 160 * s, 80 * s, 230 * s], 80, 330, fill=INK, width=7 * s)
    return hand_drawn(w, h, f)


PIGGY_SLOT = (265 / 520, 89 / 400)       # slot position as a fraction of the sprite


@lru_cache(maxsize=None)
def table_sprite(w, top_d=90, height=330):
    """Wooden table seen from the front, slightly from above. Returns the
    sprite; its tabletop back edge is at y=0, front edge at y=top_d."""
    def f(d, s):
        S = lambda *v: [x * s for x in v]
        inset = 34
        d.polygon(S(inset, 6, w - inset, 6, w - 6, top_d, 6, top_d), fill=(214, 140, 80), outline=INK, width=7 * s)
        d.line(S(inset + 30, 26, w - inset - 30, 26), fill=(232, 166, 104), width=6 * s)       # wood shine
        d.rounded_rectangle(S(6, top_d - 4, w - 6, top_d + 46), 10 * s, fill=(178, 104, 52), outline=INK, width=7 * s)
        for x in (40, w - 90):
            d.rounded_rectangle(S(x, top_d + 40, x + 50, height - 4), 10 * s, fill=(150, 86, 42), outline=INK, width=7 * s)
        d.rectangle(S(90, top_d + 46, w - 90, top_d + 66), fill=(0, 0, 0, 40))                 # shade under apron
    return hand_drawn(int(w), height, f)


@lru_cache(maxsize=None)
def tick_sprite(r=46):
    def f(d, s):
        R = r * s
        d.ellipse([4 * s, 4 * s, 2 * R - 4 * s, 2 * R - 4 * s], fill=SAVE, outline=INK, width=6 * s)
        d.line([(R * 0.5, R * 1.02), (R * 0.85, R * 1.38), (R * 1.5, R * 0.62)], fill=(255, 255, 255), width=int(R * 0.22), joint="curve")
    return hand_drawn(2 * r, 2 * r, f)


@lru_cache(maxsize=None)
def button_sprite(label, pressed=False):
    lab = text_img(label, 64, fill=(255, 255, 255))
    w, h = lab.width + 120, 150
    top = (210, 40, 50) if not pressed else (120, 120, 130)

    def f(d, s):
        d.rounded_rectangle([6 * s, 16 * s, (w - 6) * s, (h - 4) * s], 60 * s, fill=(150, 20, 30) if not pressed else (80, 80, 90), outline=INK, width=7 * s)
        d.rounded_rectangle([6 * s, (6 + (8 if pressed else 0)) * s, (w - 6) * s, (h - 18 + (8 if pressed else 0)) * s], 60 * s, fill=top, outline=INK, width=7 * s)
    im = hand_drawn(w, h, f)
    im.alpha_composite(lab, ((w - lab.width) // 2, (h - 18 - lab.height) // 2 + 4 + (8 if pressed else 0)))
    return im


@lru_cache(maxsize=None)
def bell_sprite():
    w, h = 170, 180

    def f(d, s):
        d.ellipse([70 * s, 4 * s, 100 * s, 34 * s], fill=GOLD, outline=INK, width=6 * s)
        d.pieslice([20 * s, 20 * s, 150 * s, 230 * s], 180, 360, fill=GOLD, outline=INK, width=7 * s)
        d.rectangle([20 * s, 120 * s, 150 * s, 140 * s], fill=GOLD)
        d.rounded_rectangle([8 * s, 128 * s, 162 * s, 152 * s], 12 * s, fill=GOLD_D, outline=INK, width=7 * s)
        d.ellipse([66 * s, 140 * s, 104 * s, 176 * s], fill=GOLD_D, outline=INK, width=6 * s)
        d.line([(28 * s, 120 * s), (142 * s, 120 * s)], fill=INK, width=6 * s)
    return hand_drawn(w, h, f)


@lru_cache(maxsize=None)
def phone_sprite(on):
    """Phone on a little stand: auto-save switch, salary-day calendar, arrow to piggy."""
    w, h = 300, 520

    def f(d, s):
        S = lambda *v: [x * s for x in v]
        d.rounded_rectangle(S(8, 8, w - 8, h - 8), 44 * s, fill=(50, 50, 70), outline=INK, width=7 * s)
        d.rounded_rectangle(S(26, 50, w - 26, h - 50), 22 * s, fill=(225, 245, 255))
        d.rounded_rectangle(S(110, 24, 190, 34), 5 * s, fill=(20, 20, 30))
        # mini calendar "1"
        d.rounded_rectangle(S(52, 80, 132, 170), 12 * s, fill=(255, 255, 255), outline=INK, width=4 * s)
        d.rounded_rectangle(S(52, 80, 132, 108), 12 * s, fill=NEED, outline=INK, width=4 * s)
        d.text((92 * s, 142 * s), "1", font=font(LATIN, 52 * s), fill=INK, anchor="mm")
        # arrow -> piggy coin
        d.line(S(150, 125, 196, 125), fill=INK, width=8 * s)
        d.polygon(S(196, 108, 222, 125, 196, 142), fill=INK)
        d.ellipse(S(214, 100, 264, 150), fill=GOLD, outline=INK, width=4 * s)
        # toggle
        col = SAVE if on else (190, 190, 200)
        d.rounded_rectangle(S(60, 230, 240, 310), 40 * s, fill=col, outline=INK, width=6 * s)
        kx = 200 if on else 100
        d.ellipse(S(kx - 34, 236, kx + 34, 304), fill=(255, 255, 255), outline=INK, width=5 * s)
        # list lines
        for y in (360, 400, 440):
            d.rounded_rectangle(S(60, y, 240 if y != 440 else 180, y + 16), 8 * s, fill=(170, 200, 230))
    return hand_drawn(w, h, f)


@lru_cache(maxsize=None)
def bag_sprite():
    w, h = 340, 380

    def f(d, s):
        S = lambda *v: [x * s for x in v]
        d.arc(S(90, 0, 250, 150), 180, 360, fill=INK, width=12 * s)
        d.polygon(S(30, 80, 310, 80, 330, h - 6, 10, h - 6), fill=(255, 150, 70), outline=INK, width=7 * s)
        d.polygon(S(30, 80, 310, 80, 300, 110, 40, 110), fill=(235, 120, 50), outline=INK, width=5 * s)
        d.ellipse(S(130, 180, 210, 260), fill=(255, 220, 120), outline=INK, width=5 * s)
        d.text((170 * s, 222 * s), "₹", font=font(LATIN_SYM, 44 * s), fill=(160, 80, 20), anchor="mm")
    return hand_drawn(w, h, f)


@lru_cache(maxsize=None)
def item_sprite(kind):
    def f(d, s):
        S = lambda *v: [x * s for x in v]
        if kind == "apple":
            d.ellipse(S(10, 20, 110, 116), fill=(240, 60, 60), outline=INK, width=6 * s)
            d.line(S(60, 24, 66, 4), fill=INK, width=6 * s)
            d.ellipse(S(66, 4, 100, 24), fill=(80, 190, 90), outline=INK, width=4 * s)
        elif kind == "ball":
            d.ellipse(S(8, 8, 112, 112), fill=POP, outline=INK, width=6 * s)
            d.arc(S(-40, 20, 70, 130), 290, 70, fill=(255, 255, 255), width=10 * s)
        else:  # book
            d.rounded_rectangle(S(10, 14, 110, 110), 8 * s, fill=(120, 90, 220), outline=INK, width=6 * s)
            d.rectangle(S(24, 14, 34, 110), fill=(90, 60, 180))
            d.rounded_rectangle(S(44, 40, 96, 56), 4 * s, fill=(255, 255, 255))
    return hand_drawn(120, 120, f)


@lru_cache(maxsize=None)
def stool_sprite():
    w, h = 300, 230

    def f(d, s):
        S = lambda *v: [x * s for x in v]
        d.rounded_rectangle(S(10, 6, w - 10, 56), 18 * s, fill=(77, 150, 255), outline=INK, width=7 * s)
        for x in (40, w - 80):
            d.rounded_rectangle(S(x, 50, x + 40, h - 4), 10 * s, fill=(50, 110, 210), outline=INK, width=7 * s)
        d.rounded_rectangle(S(60, 140, w - 60, 164), 8 * s, fill=(50, 110, 210), outline=INK, width=5 * s)
    return hand_drawn(w, h, f)


@lru_cache(maxsize=None)
def banner_sprite(num, label, max_w, cols):
    """Ribbon with a number chip, for tips (replaces the old chalkboard)."""
    size = 64
    txt = text_block(label, max_w - 200, size, 34, 2, fill=(255, 255, 255), stroke=7, stroke_fill=INK)
    w, h = txt.width + 190, max(150, txt.height + 60)

    def f(d, s):
        S = lambda *v: [x * s for x in v]
        d.polygon(S(20, 30, 70, 30, 70, h - 10, 20, h - 10, 46, (h + 20) / 2), fill=tuple(int(c * 0.75) for c in cols))
        d.polygon(S(w - 20, 30, w - 70, 30, w - 70, h - 10, w - 20, h - 10, w - 46, (h + 20) / 2), fill=tuple(int(c * 0.75) for c in cols))
        d.rounded_rectangle(S(50, 10, w - 50, h - 30), 24 * s, fill=cols, outline=INK, width=7 * s)
        d.ellipse(S(64, (h - 40) / 2 - 46, 156, (h - 40) / 2 + 46), fill=YELLOW_T, outline=INK, width=6 * s)
        d.text((110 * s, ((h - 40) / 2 + 2) * s), str(num), font=font(LATIN, 66 * s), fill=INK, anchor="mm")
    im = hand_drawn(w, h, f)
    im.alpha_composite(txt, (170, (h - 40 - txt.height) // 2 + 6))
    return im


@lru_cache(maxsize=None)
def tag_sprite(label, col, max_w):
    """Paper label stuck on a jar."""
    txt = fit(text_img(label, 46, fill=INK), max_w - 30)
    w, h = txt.width + 30, txt.height + 26

    def f(d, s):
        d.rounded_rectangle([3 * s, 3 * s, (w - 3) * s, (h - 3) * s], 12 * s, fill=(255, 253, 240), outline=INK, width=5 * s)
        d.rectangle([3 * s, 3 * s, (w - 3) * s, 14 * s], fill=col)
    im = hand_drawn(w, h, f)
    im.alpha_composite(txt, (15, 13 + (h - 13 - txt.height) // 2 - 4))
    return im


def bubble_sprite(text_im, col=(255, 255, 255)):
    """Speech-bubble card with a tail at the bottom (for the savings counter)."""
    w, h = text_im.width + 70, text_im.height + 50

    def f(d, s):
        S = lambda *v: [x * s for x in v]
        d.polygon(S(w / 2 - 30, h - 34, w / 2 + 30, h - 34, w / 2, h + 34), fill=col, outline=INK, width=6 * s)
        d.rounded_rectangle(S(4, 4, w - 4, h - 4), 34 * s, fill=col, outline=INK, width=7 * s)
        d.polygon(S(w / 2 - 24, h - 12, w / 2 + 24, h - 12, w / 2, h + 24), fill=col)
    im = hand_drawn(w, h + 40, f)
    im.alpha_composite(text_im, (35, 25))
    return im


@lru_cache(maxsize=64)
def sized(name, s):
    sprite = {"jar": jar_sprite, "glint": jar_glint, "piggy": piggy_sprite, "bag": bag_sprite}[name]()
    return sprite.resize((max(1, int(sprite.width * s)), max(1, int(sprite.height * s))), Image.LANCZOS)


@lru_cache(maxsize=None)
def mark_sprite(ch, col):
    return text_img(ch, 150, fill=col, stroke=9, stroke_fill=INK)


@lru_cache(maxsize=None)
def sparkle_sprite(col=(255, 255, 255), r=34):
    def f(d, s):
        pts = []
        for k in range(8):
            rr = r if k % 2 == 0 else r * 0.32
            a = math.pi / 4 * k - math.pi / 2
            pts.append(((r + rr * math.cos(a)) * s, (r + rr * math.sin(a)) * s))
        d.polygon(pts, fill=col, outline=INK, width=3 * s)
    return hand_drawn(2 * r, 2 * r, f)


# ----------------------------------------------------------- backgrounds ---
def _room_back(st):
    W, H, HZ = st.w, st.h, st.hz
    port = st.port

    def f(d, s):
        S = lambda *v: [x * s for x in v]
        # wall + wallpaper dots
        d.rectangle(S(0, 0, W, HZ), fill=(255, 226, 170))
        for y in range(40, HZ, 90):
            for x in range((y // 90 % 2) * 60, W, 120):
                d.ellipse(S(x - 7, y - 7, x + 7, y + 7), fill=(255, 212, 145))
        dado = HZ - (280 if port else 200)
        d.rectangle(S(0, dado, W, HZ), fill=(250, 200, 140))
        for x in range(30, W, 80):
            d.line(S(x, dado + 22, x, HZ - 26), fill=(238, 186, 128), width=4 * s)
        d.rectangle(S(0, dado - 10, W, dado + 10), fill=(196, 120, 70), outline=INK, width=4 * s)
        # floor with perspective tiles
        d.rectangle(S(0, HZ, W, H), fill=(240, 196, 140))
        vx = W / 2
        for x in range(-1400, W + 1400, 170):
            d.line(S(vx + (x - vx) * 0.45, HZ, x, H), fill=(218, 168, 112), width=4 * s)
        n = 6
        for k in range(1, n):
            y = HZ + (H - HZ) * (k / n) ** 1.55
            d.line(S(0, y, W, y), fill=(218, 168, 112), width=4 * s)
        d.rectangle(S(0, HZ - 24, W, HZ + 6), fill=(196, 120, 70), outline=INK, width=5 * s)   # skirting
        # window (light source: top-left)
        wx0, wy0, wx1, wy1 = (70, 290, 480, 700) if port else (110, 130, 560, 520)
        d.rounded_rectangle(S(wx0, wy0, wx1, wy1), 24 * s, fill=(255, 255, 255), outline=INK, width=8 * s)
        d.rectangle(S(wx0 + 25, wy0 + 25, wx1 - 25, wy1 - 25), fill=(140, 210, 255))
        d.ellipse(S(wx1 - 150, wy0 + 50, wx1 - 60, wy0 + 140), fill=(255, 220, 60), outline=INK, width=5 * s)
        for cx, cy in ((wx0 + 120, wy0 + 120), (wx0 + 220, wy0 + 190)):
            for dx, r in ((-40, 34), (0, 46), (44, 34)):
                d.ellipse(S(cx + dx - r, cy - r, cx + dx + r, cy + r), fill=(255, 255, 255))
        d.rectangle(S(wx0 + 25, wy1 - 120, wx1 - 25, wy1 - 25), fill=(110, 200, 110))
        d.ellipse(S(wx0 + 60, wy1 - 170, wx0 + 260, wy1 - 60), fill=(90, 185, 95))
        mx, my = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        d.line(S(mx, wy0 + 25, mx, wy1 - 25), fill=INK, width=7 * s)
        d.line(S(wx0 + 25, my, wx1 - 25, my), fill=INK, width=7 * s)
        for x0, x1 in ((wx0 - 40, wx0 + 60), (wx1 - 60, wx1 + 40)):          # curtains
            d.rounded_rectangle(S(x0, wy0 - 20, x1, wy1 + 70), 30 * s, fill=NEED, outline=INK, width=6 * s)
            d.line(S((x0 + x1) / 2, wy0, (x0 + x1) / 2, wy1 + 50), fill=(225, 80, 80), width=6 * s)
        d.rounded_rectangle(S(wx0 - 60, wy0 - 40, wx1 + 60, wy0 - 12), 12 * s, fill=(150, 90, 50), outline=INK, width=6 * s)
        # toran (mango leaves + marigolds) over the window: Indian home cue
        ty = wy0 - 70
        d.line(S(wx0 - 50, ty, wx1 + 50, ty), fill=(120, 70, 30), width=6 * s)
        k = 0
        for x in range(int(wx0 - 30), int(wx1 + 40), 44):
            if k % 2 == 0:
                d.polygon(S(x - 14, ty, x + 14, ty, x, ty + 64), fill=(70, 170, 80), outline=INK, width=3 * s)
            else:
                d.ellipse(S(x - 15, ty + 4, x + 15, ty + 34), fill=(255, 150, 20), outline=INK, width=3 * s)
            k += 1
        # clock + family photo
        cx, cy = (920, 330) if port else (1000, 170)
        d.ellipse(S(cx - 70, cy - 70, cx + 70, cy + 70), fill=(255, 255, 255), outline=INK, width=7 * s)
        for a in range(0, 360, 30):
            r = math.radians(a)
            d.ellipse(S(cx + 52 * math.cos(r) - 4, cy + 52 * math.sin(r) - 4, cx + 52 * math.cos(r) + 4, cy + 52 * math.sin(r) + 4), fill=INK)
        d.line(S(cx, cy, cx, cy - 44), fill=INK, width=7 * s)
        d.line(S(cx, cy, cx + 32, cy + 10), fill=INK, width=7 * s)
        px, py = (930, 560) if port else (760, 300)
        d.rounded_rectangle(S(px - 90, py - 70, px + 90, py + 70), 10 * s, fill=(196, 120, 70), outline=INK, width=6 * s)
        d.rectangle(S(px - 72, py - 52, px + 72, py + 52), fill=(255, 245, 225))
        for hx, hc in ((-40, (90, 60, 40)), (0, (40, 30, 30)), (40, (200, 200, 200))):
            d.ellipse(S(px + hx - 18, py - 30, px + hx + 18, py + 6), fill=SKIN, outline=INK, width=3 * s)
            d.chord(S(px + hx - 18, py - 32, px + hx + 18, py - 4), 180, 360, fill=hc)
            d.rounded_rectangle(S(px + hx - 22, py + 8, px + hx + 22, py + 52), 10 * s, fill=(POP, NEED, SAVE)[(hx + 40) // 40])
        # sofa
        if port:
            sx0, sy0, sx1, sy1 = 600, 730, 1160, 1030
        else:
            sx0, sy0, sx1, sy1 = 1180, 470, 1760, 760
        d.ellipse(S(sx0 - 20, sy1 - 30, sx1 + 20, sy1 + 30), fill=(205, 150, 100))         # sofa floor shadow
        d.rounded_rectangle(S(sx0 + 40, sy0, sx1 - 40, sy1 - 40), 40 * s, fill=(77, 150, 255), outline=INK, width=8 * s)
        d.rounded_rectangle(S(sx0, sy0 + 90, sx0 + 100, sy1), 30 * s, fill=(60, 120, 220), outline=INK, width=8 * s)
        d.rounded_rectangle(S(sx1 - 100, sy0 + 90, sx1, sy1), 30 * s, fill=(60, 120, 220), outline=INK, width=8 * s)
        d.rounded_rectangle(S(sx0 + 90, sy0 + 130, sx1 - 90, sy1), 26 * s, fill=(100, 170, 255), outline=INK, width=8 * s)
        d.rounded_rectangle(S(sx0 + 140, sy0 + 30, sx0 + 270, sy0 + 130), 26 * s, fill=WANT, outline=INK, width=6 * s)
        # plant on the floor
        plx, ply = (520, 1010) if port else (1860, 760)
        d.ellipse(S(plx - 70, ply - 12, plx + 70, ply + 14), fill=(205, 150, 100))
        for a in range(-60, 61, 30):
            r = math.radians(a - 90)
            d.ellipse(S(plx + 80 * math.cos(r) - 30, ply - 240 + 100 * math.sin(r) - 64, plx + 80 * math.cos(r) + 30, ply - 240 + 100 * math.sin(r) + 64),
                      fill=(80, 190, 90), outline=INK, width=5 * s)
        d.polygon(S(plx - 60, ply - 180, plx + 60, ply - 180, plx + 40, ply, plx - 40, ply), fill=(230, 110, 60), outline=INK, width=6 * s)
        # rug (kolam-style rings) where the boy stands
        rx, ry, rw, rh = (540, 1440, 960, 250) if port else (800, 950, 1100, 190)
        for k, c in enumerate(((210, 70, 70), (255, 196, 46), (60, 170, 160), (255, 236, 200))):
            e = k * 34
            d.ellipse(S(rx - rw / 2 + e, ry - rh / 2 + e * rh / rw, rx + rw / 2 - e, ry + rh / 2 - e * rh / rw), fill=c, outline=INK if k == 0 else None, width=5 * s)
        for k in range(12):
            a = 2 * math.pi * k / 12
            d.ellipse(S(rx + rw * 0.28 * math.cos(a) - 9, ry + rh * 0.28 * math.sin(a) - 9, rx + rw * 0.28 * math.cos(a) + 9, ry + rh * 0.28 * math.sin(a) + 9), fill=(210, 70, 70))
    img = hand_drawn(W, H, f, ss=2)
    # window light: a warm shaft on the wall and a bright patch on the floor
    light = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(light)
    if port:
        ld.polygon([(70, 700), (480, 700), (980, 1300), (420, 1300)], fill=(255, 250, 225, 34))
        ld.polygon([(200, 1100), (620, 1100), (860, 1360), (380, 1360)], fill=(255, 248, 220, 52))
    else:
        ld.polygon([(110, 520), (560, 520), (1250, 1080), (560, 1080)], fill=(255, 250, 225, 34))
        ld.polygon([(330, 780), (900, 780), (1180, 1000), (560, 1000)], fill=(255, 248, 220, 52))
    light = light.filter(ImageFilter.GaussianBlur(18))
    img.alpha_composite(light)
    # soft vignette so edges sit back
    vig = Image.new("L", (W, H), 0)
    ImageDraw.Draw(vig).rectangle([0, 0, W, H], outline=60, width=int(min(W, H) * 0.06))
    vig = vig.filter(ImageFilter.GaussianBlur(min(W, H) * 0.06))
    shade = Image.new("RGBA", (W, H), (60, 30, 10, 0))
    shade.putalpha(vig)
    img.alpha_composite(shade)
    return img.convert("RGB")


def _room_front(st):
    """Big out-of-focus leaves in a bottom corner: moves faster than the room."""
    W, H = st.w, st.h
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ox, oy = (-40, H + 20) if st.port else (-30, H + 20)
    sc = 1.25 if st.port else 1.0
    for k, (a, L) in enumerate(((-80, 420), (-55, 470), (-30, 430), (-8, 340))):
        r = math.radians(a)
        cx, cy = ox + math.cos(r) * L * sc * 0.5, oy + math.sin(r) * L * sc * 0.5
        leaf = Image.new("RGBA", (int(150 * sc), int(L * sc)), (0, 0, 0, 0))
        ImageDraw.Draw(leaf).ellipse([4, 4, leaf.width - 4, leaf.height - 4], fill=(40 + 15 * k, 130 + 10 * k, 60, 255), outline=INK, width=6)
        ImageDraw.Draw(leaf).line([leaf.width / 2, 20, leaf.width / 2, leaf.height - 20], fill=(30, 100, 45), width=6)
        leaf = leaf.rotate(-a - 90, expand=True, resample=Image.BICUBIC)
        im.alpha_composite(leaf, (int(cx - leaf.width / 2), int(cy - leaf.height / 2)))
    return im.filter(ImageFilter.GaussianBlur(3))


@lru_cache(maxsize=None)
def room(kind):
    st = SHORT if kind == "short" else LONG
    b = _room_back(st)
    return {"back": b, "back_blur": b.filter(ImageFilter.GaussianBlur(4)), "front": _room_front(st)}


@lru_cache(maxsize=None)
def burst_bg(kind, c1, c2, cx=0.5, cy=0.55, rays=22):
    st = SHORT if kind == "short" else LONG
    W, H = st.w, st.h

    def f(d, s):
        d.rectangle([0, 0, W * s, H * s], fill=c1)
        ox, oy, R = W * cx * s, H * cy * s, 2800 * s
        for i in range(rays):
            a0 = 2 * math.pi * i / rays
            a1 = a0 + math.pi / rays
            d.polygon([(ox, oy), (ox + R * math.cos(a0), oy + R * math.sin(a0)), (ox + R * math.cos(a1), oy + R * math.sin(a1))], fill=c2)
    img = hand_drawn(W, H, f, ss=1)
    glow = Image.new("L", (W, H), 0)
    r = min(W, H) * 0.42
    ImageDraw.Draw(glow).ellipse([W * cx - r, H * cy - r, W * cx + r, H * cy + r], fill=110)
    glow = glow.filter(ImageFilter.GaussianBlur(r * 0.45))
    lite = Image.new("RGBA", (W, H), (255, 255, 240, 0))
    lite.putalpha(glow)
    img.alpha_composite(lite)
    return img.convert("RGB")


@lru_cache(maxsize=None)
def card_bg(kind):
    """Cream title-card with sparkles and confetti (like the reference inserts)."""
    st = SHORT if kind == "short" else LONG
    W, H = st.w, st.h
    rnd = random.Random(7)
    keep = (W * 0.15, H * 0.3, W * 0.85, H * 0.7)

    def f(d, s):
        d.rectangle([0, 0, W * s, H * s], fill=(255, 244, 214))
        cols = [(255, 190, 60), (60, 200, 190), (255, 120, 100), (90, 150, 255)]
        for _ in range(34):
            x, y, r = rnd.randint(60, W - 60), rnd.randint(60, H - 60), rnd.randint(18, 42)
            if keep[0] < x < keep[2] and keep[1] < y < keep[3]:
                continue
            c = rnd.choice(cols)
            pts = []
            for k in range(8):
                rr = r if k % 2 == 0 else r * 0.35
                a = math.pi / 4 * k
                pts.append(((x + rr * math.cos(a)) * s, (y + rr * math.sin(a)) * s))
            d.polygon(pts, fill=c)
        for _ in range(40):
            x, y = rnd.randint(30, W - 30), rnd.randint(30, H - 30)
            if keep[0] < x < keep[2] and keep[1] < y < keep[3]:
                continue
            d.ellipse([(x - 8) * s, (y - 8) * s, (x + 8) * s, (y + 8) * s], fill=rnd.choice(cols))
    return hand_drawn(W, H, f, ss=1).convert("RGB")


# ------------------------------------------------------------- character ---
MOUTHS = {"happy": (92, 220, 175, 268), "stand": (84, 176, 148, 208), "turn": (93, 176, 154, 211)}
FULL_BODY = ("stand", "turn")


@lru_cache(maxsize=None)
def raw_pose(pose):
    return Image.open(SPRITES / f"{pose}.png").convert("RGBA")


@lru_cache(maxsize=None)
def eyes(pose):
    """Find the two eyes (white + pupil) so the boy can blink."""
    from scipy import ndimage
    a = np.asarray(raw_pose(pose)).astype(int)
    rgb, al = a[..., :3], a[..., 3]
    lim = int(a.shape[0] * (0.42 if pose in FULL_BODY else 0.75))
    white = (rgb.min(2) > 215) & (al > 200)
    white[lim:] = False
    dark = (rgb.max(2) < 45) & (al > 200)
    dark[lim:] = False
    lab, _ = ndimage.label(white)
    comps = sorted(((int((lab[sl] == i + 1).sum()), sl) for i, sl in enumerate(ndimage.find_objects(lab))),
                   key=lambda c: -c[0])[:3]
    comps = [c for c in comps if c[0] > 120]
    # teeth are also white: drop the lowest of three
    if len(comps) == 3:
        comps = sorted(comps, key=lambda c: c[1][0].start)[:2]
    dl, _ = ndimage.label(dark)
    dboxes = [sl for sl in ndimage.find_objects(dl)]
    out = []
    for _, sl in comps:
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        for ds in dboxes:   # merge the pupil that overlaps this eye (not the hair)
            dy0, dy1, dx0, dx1 = ds[0].start, ds[0].stop, ds[1].start, ds[1].stop
            if (dx1 - dx0) < 80 and dx0 < x1 and dx1 > x0 and dy0 < y1 and dy1 > y0:
                x0, y0, x1, y1 = min(x0, dx0), min(y0, dy0), max(x1, dx1), max(y1, dy1)
        out.append((x0 - 2, y0 - 2, x1 + 2, y1 + 2))
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def boy_face(pose, mouth="open", blink=False):
    """Native-size sprite with mouth state (open/mid/closed) and optional blink."""
    im = raw_pose(pose)
    out = im.copy()
    d = ImageDraw.Draw(out)
    if mouth != "open" and pose in MOUTHS:
        x0, y0, x1, y1 = MOUTHS[pose]
        mouth_px = im.crop((x0, y0, x1, y1))
        skin = im.getpixel((x0 - 6, (y0 + y1) // 2))[:3]
        if sum(skin) < 300:
            skin = SKIN
        d.rounded_rectangle([x0 - 2, y0 - 1, x1 + 2, y1 + 2], 14, fill=skin + (255,))
        mw, mh = x1 - x0, y1 - y0
        if mouth == "mid":
            sq = mouth_px.resize((mw, max(4, int(mh * 0.5))), Image.BILINEAR)
            out.alpha_composite(sq, (x0, y0 + 2))
        else:
            big = Image.new("RGBA", (mw * 4, mh * 4), (0, 0, 0, 0))
            bd = ImageDraw.Draw(big)
            bd.arc([mw * 0.4, -mh * 2.2, mw * 3.6, mh * 1.6], 30, 150, fill=INK + (255,), width=int(mh * 0.42))
            out.alpha_composite(big.resize((mw, mh), Image.LANCZOS), (x0, y0))
    if blink:
        arr = np.asarray(im).astype(int)
        for x0, y0, x1, y1 in eyes(pose):
            # skin colour = median of skin-like pixels in a ring around the eye
            X0, Y0, X1, Y1 = max(0, x0 - 10), max(0, y0 - 8), min(im.width, x1 + 10), min(im.height, y1 + 12)
            ring = arr[Y0:Y1, X0:X1].reshape(-1, 4)
            r, g, b_, a_ = ring.T
            ok = (a_ > 250) & (r > 150) & (r > g + 25) & (g > b_ + 10)
            skin = tuple(int(v) for v in np.median(ring[ok][:, :3], axis=0)) if ok.sum() > 20 else SKIN
            ew, eh = x1 - x0, y1 - y0
            k = 4                                   # draw 4x then shrink: smooth lid line
            pad = 6
            size = ((ew + pad * 2) * k, (eh + pad * 2) * k)
            mask = Image.new("L", size, 0)                 # feathered lid patch (blur the mask only)
            ImageDraw.Draw(mask).ellipse([(pad - 2) * k, (pad - 2) * k, (pad + ew + 2) * k, (pad + eh + 2) * k], fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(1.2 * k))
            lid = Image.new("RGBA", size, skin + (0,))
            lid.putalpha(mask)
            line = Image.new("RGBA", size, INK + (0,))
            ImageDraw.Draw(line).arc([pad * k, (pad + eh * 0.02) * k, (pad + ew) * k, (pad + eh * 0.72) * k], 18, 162,
                                     fill=INK + (255,), width=max(3, int(eh * 0.13)) * k)
            lid.alpha_composite(line)
            lid = lid.resize((ew + pad * 2, eh + pad * 2), Image.LANCZOS)
            out.alpha_composite(lid, (x0 - pad, y0 - pad))
    return out


@lru_cache(maxsize=None)
def light_grade(pose, mouth, blink, flip):
    """Room-light the sprite: warm tint, shade on the side away from the
    window, thin rim light on the window side. Light is always from the left
    of the screen, so this is done after flipping."""
    im = boy_face(pose, mouth, blink)
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(im).astype(np.float32)
    h, w = a.shape[:2]
    rgb = a[..., :3] * np.array([1.0, 0.975, 0.93], dtype=np.float32)
    grad = np.linspace(1.04, 0.86, w, dtype=np.float32)[None, :, None]       # lit left, shaded right
    vgrad = np.linspace(1.0, 0.94, h, dtype=np.float32)[:, None, None]       # a bit darker toward the floor
    rgb = rgb * grad * vgrad
    al = a[..., 3]
    shifted = np.zeros_like(al)
    shifted[:, 4:] = al[:, :-4]
    rim = np.clip(al - shifted, 0, 255) / 255.0                                # left edges
    rim = np.asarray(Image.fromarray((rim * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255
    rgb = rgb * (1 - rim[..., None] * 0.45) + np.array([255, 245, 215], dtype=np.float32) * rim[..., None] * 0.45
    out = np.dstack([np.clip(rgb, 0, 255), al]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


@lru_cache(maxsize=1024)
def boy_scaled(pose, mouth, blink, flip, h):
    im = light_grade(pose, mouth, blink, flip)
    s = h / im.height
    out = im.resize((max(1, int(im.width * s)), int(h)), Image.LANCZOS)
    if s > 1.3:
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=2))
    return out


@lru_cache(maxsize=256)
def cast_shadow(pose, h, flip):
    """Long soft shadow on the floor, falling right (away from the window)."""
    im = raw_pose(pose)
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    s = h / im.height
    a = im.getchannel("A").resize((max(1, int(im.width * s)), max(1, int(h * 0.26))), Image.BILINEAR)
    a = a.transpose(Image.FLIP_TOP_BOTTOM)
    shear = 1.1
    w2 = int(a.width + a.height * shear)
    a = a.transform((w2, a.height), Image.AFFINE, (1, -shear, 0, 0, 1, 0), resample=Image.BILINEAR)
    a = a.point(lambda v: int(v * 0.22)).filter(ImageFilter.GaussianBlur(7))
    out = Image.new("RGBA", a.size, SHADOW + (0,))
    out.putalpha(a)
    return out


def voice_level(tm):
    for sid, v in VOICE.items():
        st = SEG_START.get(sid)
        if st is None:
            continue
        i = int((tm - st - VOICE_LEAD) * FPS)
        if 0 <= i < len(v["env"]):
            return v["env"][i]
    return 0.0


def talk_state(local, talking, seed=0):
    if not talking:
        return "closed"
    if VOICE:
        lv = voice_level(NOW[0])
        return "open" if lv > 0.45 else "mid" if lv > 0.18 else "closed"
    k = int(local * 9) + seed
    return ("open", "mid", "open", "closed", "mid", "open", "mid", "closed")[k % 8]


def blinking(seed=0.0):
    """Blink about every 3 s (sometimes a double blink), from the global clock."""
    tm = NOW[0] + seed
    cyc = 3.1
    ph = tm % cyc
    n = int(tm // cyc)
    if ph < 0.11:
        return True
    return n % 3 == 1 and 0.22 < ph < 0.32


def draw_boy(layer, st, pose, x, foot_y, h, local, talking=False, hop=0.0, lean=0.0,
             squash=0.0, flip=False, shadow=True, seed=0):
    """Boy with feet at (x, foot_y). h = height in px. lean in degrees (+ = leans left).
    hop lifts him; squash > 0 squashes (landing), < 0 stretches (take-off)."""
    mouth = talk_state(local, talking)
    sp = boy_scaled(pose, mouth, blinking(seed * 0.7), flip, int(h) // 4 * 4)
    breathe = 1 + 0.012 * math.sin(local * 2 * math.pi / 1.5 + seed)
    sy = breathe * (1 - squash)
    sx = 1 + squash * 0.6
    if abs(sy - 1) > 0.002 or abs(sx - 1) > 0.002:
        sp = sp.resize((max(1, int(sp.width * sx)), max(1, int(sp.height * sy))), Image.BILINEAR)
    sp_h = sp.height
    sway = (1.4 * math.sin(local * 2 * math.pi / 2.3 + seed)) if talking else 0.5 * math.sin(local * 1.7 + seed)
    rot = lean + sway
    if shadow and pose in FULL_BODY:
        lift = clamp(hop / (h * 0.5))
        cs = cast_shadow(pose, int(h) // 4 * 4, flip)
        place(layer, cs, x - sp.width / 2 + cs.width / 2, foot_y - 10 + cs.height / 2, 1.0,
              alpha=1 - lift * 0.8)
        ground(layer, x, foot_y - 2, sp.width * (0.95 - 0.35 * lift), alpha=0.42 * (1 - 0.6 * lift))
    # rotate around the feet so they stay planted: rotate the sprite about its
    # centre, then move it so the bottom-centre lands back on the feet
    th = math.radians(rot)
    if abs(rot) > 0.2:
        sp = sp.rotate(rot, resample=Image.BILINEAR, expand=True)
    hh = sp_h / 2
    place(layer, sp, x - hh * math.sin(th), foot_y - hop - hh * math.cos(th), 1.0)


def walk_cycle(local, t0, t1, x0, x1, step=0.26):
    """Position, hop, lean and step events for a walk from x0 to x1."""
    if local <= t0:
        return x0, 0.0, 0.0
    p = clamp((local - t0) / (t1 - t0))
    x = x0 + (x1 - x0) * ease_out(p) if p < 1 else x1
    if p >= 1:
        return x, 0.0, 0.0
    hop = abs(math.sin((local - t0) * math.pi / step)) * 20 * (1 - p * 0.5)
    lean = -5 * (1 if x1 > x0 else -1) * (1 - p)
    return x, hop, lean


def settle(local, t_land, amt=0.07):
    """Squash-and-settle after landing (damped wobble)."""
    if local < t_land:
        return 0.0
    u = local - t_land
    return amt * math.exp(-u * 7) * math.cos(u * 22)


# --------------------------------------------------------------- camera ----
def view(img, zoom=1.0, fx=0.5, fy=0.5, shake=0.0, local=0.0):
    """Crop-and-scale the camera window out of a layer."""
    W, H = img.size
    if zoom <= 1.001 and shake == 0:
        return img
    zoom = max(1.0, zoom)
    cw, ch = W / zoom, H / zoom
    x = clamp(fx * W - cw / 2, 0, W - cw) + shake * math.sin(local * 55)
    y = clamp(fy * H - ch / 2, 0, H - ch) + shake * math.cos(local * 47)
    x, y = clamp(x, 0, W - cw), clamp(y, 0, H - ch)
    return img.resize((W, H), Image.BILINEAR, box=(x, y, x + cw, y + ch))


def compose(back_img, act, front, cam):
    """Back layer, actor layer and foreground at different camera speeds."""
    z, fx, fy, shake, local = cam
    out = view(back_img, 1 + (z - 1) * 0.72, fx, fy, shake * 0.7, local)
    out = out.copy() if out is back_img else out
    a = view(act, z, fx, fy, shake, local)
    out.paste(a, (0, 0), a)
    if front is not None:
        f = view(front, 1.03 + (z - 1) * 1.4, fx, fy, shake * 1.3, local)
        out.paste(f, (0, 0), f)
    return out


def room_shot(st, cam, draw, front=True, blur=False, ui=None):
    R = room(st.kind)
    act = Image.new("RGBA", (st.w, st.h), (0, 0, 0, 0))
    draw(act)
    out = compose(R["back_blur" if blur else "back"], act, R["front"] if front else None, cam)
    if ui:
        ui(out)
    return out


# ----------------------------------------------------------------- shots ---
class Shot:
    def __init__(self, dur, fn, cut_sfx="whoosh", events=(), cap="low"):
        self.dur, self.fn, self.cut_sfx, self.events, self.cap = dur, fn, cut_sfx, list(events), cap


SFX = []          # (time, kind)


def sfx_at(time, kind):
    SFX.append((round(time, 3), kind))


@lru_cache(maxsize=32)
def title_imgs(text, kind):
    st = SHORT if kind == "short" else LONG
    palette = ((255, 120, 90), (60, 190, 180), (90, 140, 255))
    max_w = st.w - (160 if st.port else 360)
    size = 230 if st.port else 220
    while size > 60:
        lines = wrap(text, size, max_w)
        if len(lines) <= (4 if st.port else 3) and all(text_w(l, size) <= max_w for l in lines):
            break
        size -= 8
    lines = wrap(text, size, max_w)
    return [fit(text_img(l, size, fill=palette[i % 3], stroke=12, stroke_fill=(255, 255, 255), shadow=10), max_w)
            for i, l in enumerate(lines)]


def title_card(text):
    """Big bouncy multi-colour words on the confetti card."""
    def fn(local, dur, st):
        imgs = title_imgs(text, st.kind)
        fr = card_bg(st.kind).copy()
        total = sum(i.height for i in imgs) + 20 * (len(imgs) - 1)
        y = st.h / 2 - total / 2
        for k, im in enumerate(imgs):
            sc = pop(local, 0.04 + 0.1 * k, 0.4)
            wob = 2.0 * math.sin(local * 6 + k)
            place(fr, im, st.w / 2, y + im.height / 2, sc, rot=wob)
            y += im.height + 20
        for k in range(4):                     # twinkles
            a = 2 * math.pi * k / 4 + 0.6
            place(fr, sparkle_sprite(WANT if k % 2 else (255, 255, 255)), st.w / 2 + math.cos(a) * st.w * 0.38,
                  st.h / 2 + math.sin(a) * total * 0.8, pop(local, 0.25 + 0.08 * k) * (0.8 + 0.25 * math.sin(local * 9 + k)), rot=local * 90)
        return view(fr, 1.0 + 0.05 * local / dur, 0.5, 0.5)
    return fn


def bust(st, fr, pose, local, talking=False, side=0.0, seed=0):
    """Close-up head-and-shoulders, cut by the bottom of the frame."""
    h = st.P(1000, 990)
    if isinstance(side, tuple):
        side = st.P(*side)
    x = st.w / 2 + side * st.w
    draw_boy(fr, st, pose, x, st.h + st.P(30, 40), h, local, talking=talking, shadow=False, seed=seed)


def close_up(pose, bg, marks=(), talking=False, shake_until=0.0, extra=None, side=0.0):
    def fn(local, dur, st):
        fr = burst_bg(st.kind, *bg).copy()
        if extra:
            extra(fr, st, local)
        bust(st, fr, pose, local, talking=talking, side=side)
        for k, (ch, col, dx, dy, rot) in enumerate(marks):
            place(fr, mark_sprite(ch, col), st.w / 2 + dx * st.w, st.h * dy * st.P(0.75, 1.0), pop(local, 0.12 + 0.15 * k),
                  rot=rot + 6 * math.sin(local * 5 + k))
        return view(fr, 1.0 + 0.06 * local / dur, 0.5, 0.45, shake=4 if local < shake_until else 0, local=local)
    return fn


def make_shots(seg, T):
    """Return a list of Shot for one segment, scaled to fit T seconds."""
    v = seg["visual"]
    shots = []

    def S(d, fn, sfx="whoosh", events=(), cap="low"):
        shots.append(Shot(d, fn, sfx, events, cap))

    note = note_sprite()

    if v == "title":
        heading = caps(t(seg["heading"]))
        cal = seg.get("calendar")
        cal_word = t(seg.get("calendar_label", {"en": "SALARY", "ta": "சம்பளம்", "hi": "SALARY"}))
        S(1.4, title_card(heading), "pop")

        def cal_pos(st):
            return st.P((700, 440), (1430, 330))

        def walk(local, dur, st):
            foot = st.P(1450, 985)
            x_end = st.P(360, 780)
            x, hop, lean = walk_cycle(local, 0.0, 1.0, -260, x_end)
            sq = settle(local, 1.0)
            h = st.boy_h(foot)

            def draw(act):
                if cal:
                    cx, cy = cal_pos(st)
                    place(act, drop_cached(('cal', cal[0]), calendar_sprite(cal[0], cal_word)), cx, cy, 1.0 * pop(local, 0.45, 0.35))
                draw_boy(act, st, "stand", x, foot, h, local, talking=local > 0.9, hop=hop, lean=lean, squash=sq)
                hx, hy = x + h * 0.27, foot - hop - h * 0.40
                place(act, note, hx, hy + 6 * math.sin(local * 5), 0.75 * h / 640, rot=14)
                place(act, note, hx + 14, hy - 26 + 6 * math.sin(local * 5 + 1), 0.75 * h / 640, rot=-6)
            z, fx, fy = keys(local, [(0, 1.0, 0.45, 0.6), (dur, 1.12, st.P(0.42, 0.5), st.P(0.62, 0.6))])
            return room_shot(st, (z, fx, fy, 0, local), draw)
        S(1.6, walk, events=[(0.13 + 0.26 * k, "step") for k in range(4)] + [(1.0, "boing")])
        if cal:
            def flip(local, dur, st):
                foot = st.P(1450, 985)
                bx = st.P(360, 780)
                h = st.boy_h(foot)
                win = st.P((270, 470), (330, 330))

                def draw(act):
                    cx, cy = cal_pos(st)
                    ft = 0.35
                    if local < ft:
                        c = calendar_sprite(cal[0], cal_word)
                        place(act, with_drop(c.resize((250, max(2, int(280 * (1 - local / ft)))))), cx, cy)
                    else:
                        sc = back((local - ft) / 0.35)
                        c = calendar_sprite(cal[1], cal_word)
                        place(act, with_drop(c.resize((250, max(2, int(280 * clamp(sc, 0.01, 1.3)))))), cx, cy)
                    look_left = local > 0.55
                    draw_boy(act, st, "turn", bx, foot, h, local, talking=True, flip=look_left,
                             lean=3 if look_left else 0, squash=settle(local, 0.55, 0.05))
                    hx, hy = bx + h * 0.27 * (-1 if look_left else 1), foot - h * 0.40
                    for k in range(3):     # salary notes fly out of the window
                        st_k = 0.5 + 0.22 * k
                        p = clamp((local - st_k) / 0.8)
                        if p <= 0:
                            place(act, note, hx, hy - 24 * k, 0.75 * h / 640, rot=10 - 8 * k)
                        elif p < 1:
                            nx = hx + (win[0] - hx) * p
                            ny = hy - 24 * k + (win[1] - hy) * p - math.sin(p * math.pi) * 260
                            place(act, note, nx, ny, 0.75 * h / 640 * (1 - 0.55 * p), rot=220 * p)
                z, fx, fy = keys(local, [(0, 1.18, st.P(0.55, 0.62), st.P(0.45, 0.45)), (0.5, 1.18, st.P(0.55, 0.62), st.P(0.45, 0.45)),
                                         (dur, 1.1, st.P(0.35, 0.4), st.P(0.42, 0.45))])
                return room_shot(st, (z, fx, fy, 0, local), draw, blur=True)
            S(1.6, flip, events=[(0.35, "flip")] + [(0.5 + 0.22 * k, "rustle") for k in range(3)])
        mood = seg.get("mood", "sad")
        S(max(1.0, T - sum(s.dur for s in shots)),
          close_up(mood, ((120, 170, 255), (150, 195, 255)),
                   marks=(("?", (255, 255, 255), 0.30, 0.30, 0), ("?", WANT, 0.36, 0.40, -15)), shake_until=0.3),
          cap="high")

    elif v == "split":
        parts = seg["parts"]
        labels = t(seg["labels"])
        cols = [NEED, WANT, SAVE, POP]
        S(1.3, title_card("-".join(str(p) for p in parts)), "pop")
        jar, glint, env = jar_sprite(), jar_glint(), envelope_sprite()
        n_notes = [max(1, round(p / 10)) for p in parts]
        order = [(j, k) for j, n in enumerate(n_notes) for k in range(n)]
        fly = 0.3

        def layout(st):
            if st.port:
                return dict(jx=[215, 540, 865], jar_b=1100, jsc=0.84, tw=1040, tx=540, ty=1068, boy=(540, 1090), env=(540, 1152), tagw=240)
            return dict(jx=[930, 1270, 1610], jar_b=785, jsc=0.80, tw=1300, tx=1270, ty=752, boy=(420, 975), env=(1270, 838), tagw=300)

        def table_scene(local, dur, st, zoomed=False):
            L = layout(st)
            tab = table_sprite(L["tw"], st.P(110, 100), st.P(420, 330))
            js = L["jsc"]
            jh = 360 * js

            def draw(act):
                bx, bf = L["boy"]
                if not st.port:                     # boy stands beside the table
                    draw_boy(act, st, "turn", bx, bf, st.boy_h(bf), local, talking=not zoomed)
                else:                               # boy behind the table, legs hidden
                    draw_boy(act, st, "stand", bx, bf, st.boy_h(bf), local, talking=not zoomed, shadow=False)
                ground(act, L["tx"], L["ty"] + tab.height - 6, L["tw"] * 1.02, alpha=0.32, flat=0.08)
                place(act, tab, L["tx"], L["ty"] + tab.height / 2)
                filled = [0] * len(parts)
                flying = []
                for idx, (j, k) in enumerate(order):
                    st_t = 0.55 + idx * fly
                    p = (local - st_t) / 0.45
                    if zoomed or p >= 1:
                        filled[j] += 1
                    elif p > 0:
                        ex, ey = L["env"]
                        x = ex + (L["jx"][j] - ex) * p
                        y = ey - 40 + (L["jar_b"] - jh - 30 - (ey - 40)) * p - math.sin(p * math.pi) * st.P(260, 200)
                        flying.append((x, y, p))
                d = ImageDraw.Draw(act)
                for j, x in enumerate(L["jx"]):
                    ground(act, x, L["jar_b"] - 4, 230 * js, alpha=0.4)
                    level = filled[j] / n_notes[j] * parts[j] / max(parts)
                    if level > 0:
                        top = L["jar_b"] - 8 * js - int((jh - 90 * js) * level)
                        d.rounded_rectangle([x - 112 * js, top, x + 112 * js, L["jar_b"] - 10 * js], int(36 * js), fill=cols[j % 4])
                        for q in range(filled[j]):     # notes visible inside
                            ny = L["jar_b"] - 30 * js - q * 22 * js
                            if ny > top + 10:
                                d.rounded_rectangle([x - 70 * js, ny - 10 * js, x + 70 * js, ny + 6 * js], 5, fill=(124, 214, 140), outline=INK, width=2)
                    place(act, sized("jar", js), x, L["jar_b"], anchor="bottom")
                    place(act, sized("glint", js), x, L["jar_b"], anchor="bottom")
                    tag = tag_sprite(labels[j], cols[j % 4], L["tagw"])
                    place(act, tag, x, L["jar_b"] - jh * 0.62, 1.0 if zoomed else pop(local, 0.15 + 0.12 * j), rot=-3 + 3 * j)
                    done_t = 0.55 + (sum(n_notes[:j + 1]) - 1) * fly + 0.45
                    if zoomed or local > done_t:
                        pc = text_img(f"{parts[j]}%", st.P(84, 80), fill=(255, 255, 255), stroke=9, stroke_fill=INK, shadow=5)
                        sc = 1.0 + (0.06 * math.sin(local * 6 + j) if zoomed else 0)
                        place(act, pc, x, L["jar_b"] - jh * 0.25, sc if zoomed else pop(local, done_t))
                ex, ey = L["env"]
                place(act, env, ex, ey, st.P(0.8, 0.72) * (1 + 0.04 * math.sin(local * 14) if 0.5 < local < 0.55 + len(order) * fly else 1))
                for x, y, p in flying:
                    place(act, note, x, y, st.P(0.55, 0.5) * (1 - 0.35 * p), rot=360 * p)
            if zoomed:
                z, fx, fy = keys(local, [(0, st.P(1.16, 1.3), 0.5, st.P(0.54, 0.66)), (dur, st.P(1.2, 1.38), 0.5, st.P(0.54, 0.66))])
            else:
                z, fx, fy = keys(local, [(0, st.P(1.04, 1.08), 0.5, st.P(0.5, 0.6)), (dur, st.P(1.1, 1.16), st.P(0.5, 0.6), st.P(0.5, 0.62))])
            return room_shot(st, (z, fx, fy, 0, local), draw, front=not zoomed, blur=zoomed)
        coin_time = 0.55 + len(order) * fly + 0.5
        S(max(3.0, coin_time + 0.4), table_scene,
          events=[(0.5, "pop")] + [(0.55 + i * fly, "rustle") for i in range(len(order))] + [(0.55 + i * fly + 0.45, "coin") for i in range(len(order))])

        def jars_close(local, dur, st):
            return table_scene(local, dur, st, zoomed=True)

        def think_extra(fr, st, local):
            for j in range(len(parts)):
                pc = text_img(f"{parts[j]}%", 130, fill=cols[j % 4], stroke=12, stroke_fill=INK, shadow=8)
                if st.port:
                    px, py = 200 + 340 * j, 470 + 70 * (j % 2)
                else:
                    px, py = 1250 + 230 * j, 300 + 170 * j
                place(fr, pc, px, py + 8 * math.sin(local * 4 + j), pop(local, 0.15 + 0.15 * j), rot=-8)
        rest = T - sum(s.dur for s in shots)
        think = close_up("think", ((255, 200, 80), (255, 220, 120)), extra=think_extra, side=0)
        if rest > 3.0:
            S(rest - 1.6, jars_close)
            S(1.6, think, cap="high")
        else:
            S(max(1.0, rest), think, cap="high")

    elif v == "number":
        value = str(seg["value"])
        digits = int("".join(ch for ch in value if ch.isdigit()) or 0)
        prefix = value[: next((i for i, ch in enumerate(value) if ch.isdigit()), 0)]
        sub = t(seg.get("caption", ""))
        total = seg.get("total")
        piggy, coin, stack = piggy_sprite(), coin_sprite(40), note_stack(8)
        S(1.3, close_up("wow", ((255, 170, 60), (255, 200, 90)),
                        marks=(("!", NEED, 0.32, 0.30, 10), ("!", WANT, -0.33, 0.32, -12)), shake_until=0.35),
          "pop", cap="high")

        def count(local, dur, st):
            if st.port:
                L = dict(boy=(210, 1480), tx=650, ty=1080, tw=860, stack=(460, 1112), pig=(820, 1118), psc=0.6,
                         bub=(780, 650), sub=(540, 450))
            else:
                L = dict(boy=(330, 990), tx=1220, ty=760, tw=1300, stack=(880, 790), pig=(1480, 795), psc=0.72,
                         bub=(1430, 360), sub=(1430, 180))
            tab = table_sprite(L["tw"], st.P(110, 100), st.P(420, 330))
            pw, ph = 520 * L["psc"], 400 * L["psc"]
            n = int(round(digits * ease_out((local - 0.3) / (dur * 0.6))))   # lands exactly on the value
            s_ = f"{n:,}"
            if digits >= 100000 and n >= 100000:   # Indian grouping for lakhs
                s_ = f"{n // 100000},{n // 1000 % 100:02d},{n % 1000:03d}"
            num = text_img(prefix + s_, st.P(120, 130), fill=SAVE, stroke=10, stroke_fill=INK, shadow=6)

            def draw(act):
                ground(act, L["tx"], L["ty"] + tab.height - 6, L["tw"] * 1.02, alpha=0.32, flat=0.08)
                place(act, tab, L["tx"], L["ty"] + tab.height / 2)
                sx, sy = L["stack"]
                ground(act, sx, sy - 4, 230, alpha=0.4)
                gone = int(clamp((local - 0.3) / (dur * 0.6)) * 2.99)   # notes leave the stack as coins fly
                place(act, note_stack(8 - min(2, gone)), sx, sy, st.P(0.9, 1.0), anchor="bottom")
                if total:
                    tl = text_img(str(total), st.P(56, 60), fill=(255, 255, 255), stroke=7, stroke_fill=INK)
                    place(act, tl, sx, sy - stack.height * st.P(0.9, 1.0) - 40, pop(local, 0.05))
                px, py = L["pig"]
                ground(act, px, py - 4, pw * 0.9, alpha=0.4)
                wig = 2.5 * math.sin(local * 18) * math.exp(-((local % 0.45) * 6)) if local > 0.6 else 0
                place(act, piggy, px, py, L["psc"], rot=wig, anchor="bottom")
                slot = (px - pw / 2 + pw * PIGGY_SLOT[0], py - ph + ph * PIGGY_SLOT[1])
                for k in range(6):
                    st_k = 0.3 + k * 0.3
                    p = clamp((local - st_k) / 0.45)
                    if 0 < p < 1:
                        x = sx + (slot[0] - sx) * p
                        y = sy - 120 + (slot[1] - 20 - (sy - 120)) * p - math.sin(p * math.pi) * st.P(220, 180)
                        place(act, coin, x, y, 0.9, rot=300 * p)
                bub = bubble_sprite(num)
                place(act, bub, L["bub"][0], L["bub"][1], pop(local, 0.25, 0.3) * (1 + 0.03 * math.sin(local * 7)))
                if sub:
                    si = text_block(sub, st.P(900, 700), 56, 34, 2, fill=(255, 255, 255), stroke=7, stroke_fill=INK)
                    place(act, si, L["sub"][0], L["sub"][1], pop(local, 0.5))
                bx, bf = L["boy"]
                draw_boy(act, st, "turn", bx, bf, st.boy_h(bf), local, talking=True, hop=abs(math.sin(local * 6)) * 14 if 1.9 < local < 2.6 else 0)
            z, fx, fy = keys(local, [(0, 1.0, 0.5, 0.5), (dur, st.P(1.08, 1.1), st.P(0.62, 0.66), st.P(0.48, 0.5))])
            return room_shot(st, (z, fx, fy, 0, local), draw)
        S(2.8, count, events=[(0.3 + k * 0.3 + 0.45, "coin") for k in range(6)])
        badge = text_img(value, 150, fill=SAVE, stroke=12, stroke_fill=INK, shadow=8)

        def happy_extra(fr, st, local):
            bx, by = st.P((540, 470), (1480, 430))
            place(fr, badge, bx, by, pop(local, 0.2) * (1 + 0.04 * math.sin(local * 6)), rot=-6)
            for k in range(5):
                a = 2 * math.pi * k / 5 + local * 1.5
                place(fr, coin_sprite(34), bx + math.cos(a) * 300, by + math.sin(a) * 120, pop(local, 0.3 + 0.06 * k))
        S(max(1.0, T - 4.1), close_up("happy", ((255, 210, 80), (255, 230, 130)), talking=True,
                                      extra=happy_extra, side=(0.0, -0.17)), cap="high")

    elif v == "bullets":
        items = t(seg["items"])
        icons = seg.get("icons") or ["piggy", "phone", "bag"]
        n = len(items)
        per = T / n
        cols_b = [(255, 120, 90), (77, 150, 255), (88, 190, 102), (180, 110, 230)]

        def vignette(k):
            icon = icons[k % len(icons)]

            def fn(local, dur, st):
                foot = st.P(1470, 975)
                bx = st.P(250, 430)
                h = st.boy_h(foot)
                px, pf = st.P((770, 1400), (1300, 950))
                ban = banner_sprite(k + 1, items[k], st.P(980, 1300), cols_b[k % 4])

                def draw(act):
                    jump = abs(math.sin(local * 8)) * 18 if local < 0.38 else 0
                    draw_boy(act, st, "turn", bx, foot, h, local, talking=True, hop=jump, squash=settle(local, 0.4, 0.05), seed=k)
                    if icon == "piggy":
                        sc = st.P(0.66, 0.8)
                        pw, ph = 520 * sc, 400 * sc
                        ground(act, px, pf - 4, pw * 0.9)
                        hit = 0.95
                        wig = 4 * math.sin((local - hit) * 20) * math.exp(-(local - hit) * 5) if local > hit else 0
                        place(act, piggy_sprite(), px, pf, sc * (1 + (0.06 * math.exp(-(local - hit) * 8) if local > hit else 0)), rot=wig, anchor="bottom")
                        slot = (px - pw / 2 + pw * PIGGY_SLOT[0], pf - ph + ph * PIGGY_SLOT[1])
                        p = clamp((local - 0.45) / 0.5)
                        if p < 1:
                            place(act, coin_sprite(46), slot[0], slot[1] - 300 * (1 - ease_io(p)) - 30, 1.0, rot=180 * p)
                        place(act, text_img("+₹", 70, fill=SAVE, stroke=8, stroke_fill=INK), slot[0] + 120, slot[1] - 120 - 40 * clamp(local - hit), pop(local, hit))
                    elif icon == "phone":
                        stool = stool_sprite()
                        ssc = st.P(0.95, 0.9)
                        ground(act, px, pf - 4, 300 * ssc)
                        place(act, stool, px, pf, ssc, anchor="bottom")
                        on = local > 0.8
                        ph_ = drop_cached(('phone', on), phone_sprite(on), 8, 10, 0.3, 5)
                        place(act, ph_, px, pf - 220 * ssc + 30, st.P(0.9, 0.85) * pop(local, 0.1, 0.4), rot=-4, anchor="bottom")
                        if on:
                            place(act, tick_sprite(44), px + 150, pf - 220 * ssc - 420, pop(local, 0.85))
                    else:
                        sc = st.P(0.85, 0.85)
                        ground(act, px, pf - 4, 340 * sc)
                        place(act, bag_sprite(), px, pf, sc, anchor="bottom")
                        for q, kind in enumerate(("apple", "ball", "book")):
                            st_q = 0.4 + q * 0.35
                            p = clamp((local - st_q) / 0.4)
                            if 0 < p < 1:
                                place(act, item_sprite(kind), px - 60 + 60 * q, pf - 380 * sc - 260 * (1 - ease_io(p)), 1.0, rot=90 * p)
                            elif p >= 1:
                                place(act, item_sprite(kind), px - 60 + 60 * q, pf - 330 * sc - 6 * q, 0.9, rot=-10 + 10 * q)
                        place(act, bag_sprite().crop((0, 110, 340, 380)), px, pf, sc, anchor="bottom")
                def ui(fr):
                    place(fr, ban, st.w / 2 if st.port else 1180, st.P(370, 130), pop(local, 0.05, 0.35))
                z, fx, fy = keys(local, [(0, 1.0, 0.5, 0.5), (dur, st.P(1.07, 1.08), st.P(0.55, 0.6), st.P(0.62, 0.62))])
                return room_shot(st, (z, fx, fy, 0, local), draw, ui=ui)
            ev = {"piggy": [(0.95, "coin")], "phone": [(0.8, "click"), (0.85, "ding")],
                  "bag": [(0.8, "pop"), (1.15, "pop"), (1.5, "pop")]}[icon]
            return fn, ev

        nod = close_up("happy", ((120, 210, 200), (150, 228, 218)), talking=True)
        for k in range(n):
            fn, ev = vignette(k)
            if per > 2.4 and k < n - 1:
                S(per - 0.9, fn, "pop" if k else "whoosh", events=ev)
                S(0.9, nod, cap="high")
            else:
                S(per, fn, "pop", events=ev)

    elif v == "outro":
        note_txt = caps(t(seg["heading"]))
        btn, btn_done, bell = button_sprite("SUBSCRIBE"), button_sprite("SUBSCRIBED", pressed=True), bell_sprite()

        def sub(local, dur, st):
            foot = st.P(1470, 985)
            bx = st.P(300, 430)
            h = st.boy_h(foot)
            hop = abs(math.sin(local * 5)) * st.P(70, 60) if local < 1.26 else 0
            press = 1.7
            note_img = text_block(note_txt, st.P(820, 900), 50, 30, 2, fill=INK)
            card = Image.new("RGBA", (note_img.width + 70, note_img.height + 50), (0, 0, 0, 0))
            ImageDraw.Draw(card).rounded_rectangle([3, 3, card.width - 3, card.height - 3], 22, fill=(255, 253, 240), outline=INK, width=5)
            card.alpha_composite(note_img, (35, 25))
            bpos = st.P((560, 560), (1180, 380))
            bellp = st.P((560 + btn.width * 0.6 + 50, 400), (1180 + btn.width * 0.75 + 140, 380))
            cpos = st.P((600, 790), (1250, 650))

            def draw(act):
                draw_boy(act, st, "stand", bx, foot, h, local, talking=local > 1.3, hop=hop,
                         squash=settle(local, 1.26, 0.08) - (0.05 if 0 < hop < 20 and local < 1.26 else 0))
                b = btn if local < press else btn_done
                sc = pop(local, 0.4, 0.4)
                if press - 0.08 < local < press + 0.12:
                    sc *= 0.92
                place(act, drop_cached(('btn', local < press), b), bpos[0], bpos[1], st.P(1.15, 1.5) * sc)
                if press <= local < press + 0.5:          # tap ripple
                    r = (local - press) / 0.5
                    ring = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
                    ImageDraw.Draw(ring).ellipse([150 - 140 * r, 150 - 140 * r, 150 + 140 * r, 150 + 140 * r],
                                                 outline=(255, 255, 255, int(255 * (1 - r))), width=10)
                    place(act, ring, bpos[0] + 60, bpos[1] + 20)
                rot = 18 * math.sin(local * 18) * math.exp(-(local - 2.0) * 2) if local > 2.0 else 0
                place(act, drop_cached('bell', bell), bellp[0], bellp[1], st.P(1.0, 1.3) * pop(local, 0.9), rot=rot)
                place(act, drop_cached(('card', st.kind), card, 8, 10, 0.25, 5), cpos[0], cpos[1], pop(local, 0.6))
            z, fx, fy = keys(local, [(0, 1.0, 0.5, 0.5), (dur, 1.05, 0.5, 0.5)])
            return room_shot(st, (z, fx, fy, 0, local), draw)
        S(T, sub, "pop", events=[(0.25, "boing"), (1.26, "boing"), (1.7, "click"), (2.0, "ding")])

    else:
        raise SystemExit(f"unknown visual {v}")

    # fit to T exactly: stretch/squeeze shot lengths proportionally
    tot = sum(s.dur for s in shots)
    for s in shots:
        s.dur *= T / tot
    return shots



# -------------------------------------------------------------- captions ---
def caption_chunks(seg, start, T):
    """Short chunks with per-word times: [(t0, t1, [(w, wt0, wt1), ...]), ...]."""
    if not SCRIPT.get("captions", True) or "say" not in seg:
        return []
    text = caps(t(seg["say"]) if LANG != "hi" else (seg["say"].get("hinglish") or seg["say"].get("en")))
    per = 3 if TEXT_LANG == "en" else 2
    v = VOICE.get(seg["id"])
    if v and v["words"]:
        timed = [(caps(w_), start + VOICE_LEAD + a, start + VOICE_LEAD + b) for a, b, w_ in v["words"]]
    else:
        words = text.split()
        lead, tail = 0.25, 0.35
        span = max(0.5, T - lead - tail)
        tot = sum(len(w_) + 2 for w_ in words)
        timed, t0 = [], start + lead
        for w_ in words:
            d = span * (len(w_) + 2) / tot
            timed.append((w_, t0, t0 + d))
            t0 += d
    out, cur = [], []
    for item in timed:
        cur.append(item)
        if len(cur) >= per or item[0][-1:] in ".?!,:;":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    chunks = []
    for k, ws in enumerate(out):
        t0 = ws[0][1]
        t1 = out[k + 1][0][1] if k + 1 < len(out) else min(start + T, ws[-1][2] + 0.5)
        chunks.append((t0, t1, ws))
    return chunks


@lru_cache(maxsize=512)
def caption_img(words, active, size, max_w):
    """Words laid out on up to 2 lines; the word being spoken is yellow."""
    while True:
        space = text_w(" ", size) * 1.1
        ims = [text_img(w_, size, fill=YELLOW_T if i == active else (255, 255, 255), stroke=max(6, size // 9),
                        stroke_fill=(0, 0, 0), shadow=max(3, size // 16)) for i, w_ in enumerate(words)]
        lines, cur, cw = [], [], 0
        for im in ims:
            if cur and cw + space + im.width > max_w:
                lines.append(cur)
                cur, cw = [], 0
            cur.append(im)
            cw += im.width + (space if len(cur) > 1 else 0)
        lines.append(cur)
        if len(lines) <= 2 or size <= 40:
            break
        size -= 6
    lh = max(i.height for i in ims)
    widths = [sum(i.width for i in l) + space * (len(l) - 1) for l in lines]
    out = Image.new("RGBA", (int(max(widths)) + 4, lh * len(lines) + 10 * (len(lines) - 1)), (0, 0, 0, 0))
    for li, l in enumerate(lines):
        x = (out.width - widths[li]) / 2
        for im in l:
            out.alpha_composite(im, (int(x), li * (lh + 10) + (lh - im.height) // 2))
            x += im.width + space
    return fit(out, max_w)


def draw_caption(frame, st, cap, tm, zone, zone_t):
    t0, t1, ws = cap
    active = max([i for i, (_, a, _) in enumerate(ws) if a <= tm] or [0])
    words = tuple(w_ for w_, _, _ in ws)
    if st.port:
        img = caption_img(words, active, 92, 860)
        y_low, y_high = 1590, 800
        y = lerp(y_high, y_low, zone_t)
        x = 520                       # a touch left of centre: right side has the Shorts buttons
    else:
        img = caption_img(words, active, 84, 1500)
        y, x = st.h - 70 - img.height / 2, st.w / 2
    sc = 0.85 + 0.15 * pop(tm - t0, 0.0, 0.18)
    place(frame, img, x, y, sc)


# ---------------------------------------------------------------- audio ----
def synth_sfx(length, path, rate=44100):
    n = int((length + 0.5) * rate)
    track = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(3)

    def tone(dur, f0, f1, vol):
        tt = np.arange(int(dur * rate)) / rate
        f = np.linspace(f0, f1, len(tt))
        ph = 2 * np.pi * np.cumsum(f) / rate
        env = np.exp(-tt / (dur * 0.35))
        return (np.sin(ph) * env * vol).astype(np.float32)

    def noise(dur=0.35, vol=0.18, k=40):
        m = int(dur * rate)
        nz = rng.standard_normal(m).astype(np.float32)
        nz = np.convolve(nz, np.ones(k) / k, mode="same")
        env = np.sin(np.linspace(0, np.pi, m)) ** 2
        return nz * env * vol * 3

    def boing():
        tt = np.arange(int(0.35 * rate)) / rate
        f = 220 + 260 * (tt / 0.35) + 40 * np.sin(2 * np.pi * 18 * tt)
        ph = 2 * np.pi * np.cumsum(f) / rate
        return (np.sin(ph) * np.exp(-tt / 0.15) * 0.25).astype(np.float32)

    bank = {
        "pop": lambda: tone(0.12, 500, 1300, 0.30),
        "coin": lambda: np.concatenate([tone(0.08, 1800, 1800, 0.22), tone(0.25, 2400, 2400, 0.22)]),
        "whoosh": noise,
        "flip": lambda: noise(0.18, 0.25),
        "rustle": lambda: noise(0.16, 0.12, 6),
        "step": lambda: tone(0.07, 160, 90, 0.35),
        "boing": boing,
        "click": lambda: tone(0.06, 900, 400, 0.35),
        "ding": lambda: tone(0.6, 1320, 1320, 0.25) + tone(0.6, 1980, 1980, 0.08),
    }
    for tm, kind in SFX:
        snd = bank[kind]()
        i = int(tm * rate)
        if 0 <= i < n:
            seg = snd[: n - i]
            track[i:i + len(seg)] += seg
    track = np.clip(track, -1, 1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes((track * 32767).astype(np.int16).tobytes())


# ----------------------------------------------------------------- main ----
def series_title():
    st = SCRIPT.get("series_title")
    if st:
        return caps(t(st))
    first = next((s for s in SCRIPT["segments"] if "heading" in s), None)
    return caps(t(first["heading"]) if first else SCRIPT.get("title", ""))


def short_header():
    """Compact top pill for the Short: coin logo, yellow EP chip, title."""
    ep = text_img(f"EP {SCRIPT.get('ep', 0)}", 44, fill=INK)
    chip = Image.new("RGBA", (ep.width + 36, ep.height + 22), (0, 0, 0, 0))
    ImageDraw.Draw(chip).rounded_rectangle([0, 0, chip.width - 1, chip.height - 1], 18, fill=YELLOW_T)
    chip.alpha_composite(ep, (18, 11))
    logo = coin_sprite(42)
    title = text_block(series_title(), 900 - chip.width - logo.width - 70, 50, 30, 2, fill=(255, 255, 255), stroke=4, stroke_fill=(0, 0, 0))
    w = logo.width + chip.width + title.width + 80
    h = max(title.height, chip.height, logo.height) + 36
    pill = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(pill).rounded_rectangle([0, 0, w - 1, h - 1], h // 2 if h < 120 else 40, fill=(20, 12, 8, 150))
    x = 18
    pill.alpha_composite(logo, (x, (h - logo.height) // 2))
    x += logo.width + 16
    pill.alpha_composite(chip, (x, (h - chip.height) // 2))
    x += chip.width + 18
    pill.alpha_composite(title, (x, (h - title.height) // 2))
    return pill


RENDER = {}       # shared with worker processes (fork)


def ffmpeg_in(path, w, h):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                             "-crf", "19", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)


def render_range(job):
    """Render frames [a, b) of every target into part files (one process)."""
    idx, a, b = job
    R = RENDER
    procs = [ffmpeg_in(f"{raw}.part{idx:02d}.mp4", st.w, st.h) for st, raw, _ in R["targets"]]
    si = 0
    tl = R["timeline"]
    for fi in range(a, b):
        tm = fi / FPS
        while si + 1 < len(tl) and tl[si + 1][0] <= tm:
            si += 1
        g0, sh = tl[si]
        local = tm - g0
        NOW[0] = tm
        cap = next((c for c in R["caps"] if c[0] <= tm < c[1]), None)
        for (st, _, _), p in zip(R["targets"], procs):
            frame = sh.fn(local, sh.dur, st)
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            if local < 0.12:                 # punch-in on every cut
                frame = view(frame, 1.0 + 0.05 * (1 - local / 0.12))
            if st.port:
                place(frame, R["header"], st.w / 2 - 20, 150)
            else:
                place(frame, R["logo"], st.w - 90, 80)
            if cap:
                draw_caption(frame, st, cap, tm, sh.cap, R["zone"][fi])
            p.stdin.write(frame.tobytes())
        if (fi - a) % 150 == 0:
            print(f"[part {idx}] frame {fi - a}/{b - a}", flush=True)
    for p in procs:
        p.stdin.close()
        p.wait()
    return idx


def main():
    import multiprocessing as mp
    out_dir = Path("out/media/cartoon")
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline, caps_list, g = [], [], 0.0
    for seg in SCRIPT["segments"]:
        T = seg_time(seg)
        SEG_START[seg["id"]] = g
        for sh in make_shots(seg, T):
            timeline.append((g, sh))
            if sh.cut_sfx and g > 0:
                sfx_at(g + 0.01, sh.cut_sfx)
            for lt, kind in sh.events:
                if lt < sh.dur:
                    sfx_at(g + lt, kind)
            g += sh.dur
        caps_list += caption_chunks(seg, g - T, T)
    length = g if PREVIEW is None else min(g, PREVIEW)
    nframes = int(round(length * FPS))

    # caption height per frame: 1 = low, 0 = high (above the face in close-ups)
    zone, si = [], 0
    for fi in range(nframes):
        tm = fi / FPS
        while si + 1 < len(timeline) and timeline[si + 1][0] <= tm:
            si += 1
        zone.append(1.0 if timeline[si][1].cap == "low" else 0.0)   # jump with the cut, never slide over a face

    targets = []
    if ONLY in (None, "long"):
        targets.append((LONG, out_dir / f"{LANG}_v.mp4", out_dir / f"{LANG}.mp4"))
    if ONLY in (None, "short"):
        targets.append((SHORT, out_dir / f"{LANG}_short_v.mp4", out_dir / f"{LANG}_short.mp4"))
    RENDER.update(timeline=timeline, caps=caps_list, zone=zone, targets=targets,
                  header=short_header(), logo=coin_sprite(56))

    jobs_n = int(os.environ.get("RENDER_JOBS", 0)) or max(1, min(8, os.cpu_count() or 1))
    step = math.ceil(nframes / jobs_n)
    jobs = [(k, k * step, min(nframes, (k + 1) * step)) for k in range(jobs_n) if k * step < nframes]
    print(f"{nframes} frames, {len(jobs)} parallel part(s)", flush=True)
    if len(jobs) == 1:
        render_range(jobs[0])
    else:
        with mp.get_context("fork").Pool(len(jobs)) as pool:
            pool.map(render_range, jobs)

    wav = out_dir / f"{LANG}_sfx.wav"
    synth_sfx(length, wav)
    for _, raw, final in targets:
        lst = out_dir / f"{raw.stem}_parts.txt"
        lst.write_text("".join(f"file '{Path(f'{raw}.part{k:02d}.mp4').resolve()}'\n" for k, _, _ in jobs))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)], check=True)
        for k, _, _ in jobs:
            Path(f"{raw}.part{k:02d}.mp4").unlink()
        lst.unlink()
    print("wrote", ", ".join(str(f) for _, _, f in targets), f"{length:.1f}s")


if __name__ == "__main__":
    main()
