"""Cartoon renderer: turns episodes/<ep>/script.yaml into a kids'-cartoon style
video, in the look of the reference Short (see README "House style").

Every segment is cut into quick shots (1-2.5 s each): title-card inserts, wide
shots of the boy in a drawn room, close-up reactions, props that move (cash,
calendar, jars, piggy bank, chalkboard). The boy "talks" (mouth flaps) while
his line is on screen. Captions are short all-caps chunks, like the reference.

Writes two videos from the same shots:
  out/media/cartoon/<lang>.mp4        16:9 long video (1920x1080)
  out/media/cartoon/<lang>_short.mp4  9:16 Short/Reel (1080x1920): black frame,
                                      yellow series title, 16:9 panel, caption
Both carry light cartoon sound effects (pops, whooshes, coin dings).

Usage:  python pipeline/cartoon.py <episode_dir> <lang> [--preview SECONDS]
Env:    DURATIONS=out/durations.json (Hindi voice lengths, optional)
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

W, H, FPS = 1920, 1080, 30
SW, SH = 1080, 1920                     # Short canvas
PANEL_Y = 650                           # top of the 16:9 panel inside the Short
SPRITES = Path("assets/characters/sprites")

INK = (43, 27, 23)
GOLD, GOLD_D = (255, 196, 46), (214, 138, 0)
NEED, WANT, SAVE, POP = (255, 107, 107), (255, 211, 61), (88, 204, 102), (77, 150, 255)
YELLOW_T = (255, 214, 10)
SKIN = (238, 160, 106)

# ---------------------------------------------------------------- script ---
EP = Path(sys.argv[1] if len(sys.argv) > 1 else "episodes/week00-test")
LANG = sys.argv[2] if len(sys.argv) > 2 else "ta"
PREVIEW = float(sys.argv[sys.argv.index("--preview") + 1]) if "--preview" in sys.argv else None
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
NOW = [0.0]               # current video time, for lip sync


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
        elif ch == "₹":
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
        if len(lines) <= max_lines:
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
    return out


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


# --------------------------------------------------------------- drawing ---
def hand_drawn(w, h, fn, ss=2):
    """Draw at 2x with fn(draw, scale) then shrink: smooth bold outlines."""
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    fn(ImageDraw.Draw(im), ss)
    return im.resize((w, h), Image.LANCZOS)


def place(frame, sprite, cx, cy, scale=1.0, rot=0.0, anchor="center", alpha=1.0):
    """Paste sprite with its center (or bottom) at cx, cy."""
    if scale <= 0.01 or sprite is None:
        return
    sp = sprite
    if abs(scale - 1) > 0.01:
        sp = sp.resize((max(1, int(sp.width * scale)), max(1, int(sp.height * scale))), Image.BILINEAR)
    if abs(rot) > 0.3:
        sp = sp.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 0.99:
        a = sp.getchannel("A").point(lambda v: int(v * alpha))
        sp = sp.copy()
        sp.putalpha(a)
    x = int(cx - sp.width / 2)
    y = int(cy - sp.height / 2) if anchor == "center" else int(cy - sp.height)
    frame.paste(sp, (x, y), sp)


def coin_sprite(r=46):
    def f(d, s):
        R = r * s
        d.ellipse([4 * s, 4 * s, 2 * R - 4 * s, 2 * R - 4 * s], fill=GOLD, outline=INK, width=5 * s)
        d.ellipse([R * 0.32, R * 0.32, R * 1.68, R * 1.68], outline=GOLD_D, width=4 * s)
        d.text((R, R * 1.06), "₹", font=font(LATIN_SYM, int(R * 1.05)), fill=(122, 75, 0), anchor="mm")
    return hand_drawn(2 * r, 2 * r, f)


def note_sprite():
    w, h = 220, 110

    def f(d, s):
        d.rounded_rectangle([4 * s, 4 * s, (w - 4) * s, (h - 4) * s], 12 * s, fill=(124, 214, 140), outline=INK, width=5 * s)
        d.rounded_rectangle([18 * s, 16 * s, (w - 18) * s, (h - 16) * s], 8 * s, outline=(50, 140, 70), width=3 * s)
        d.ellipse([(w / 2 - 30) * s, (h / 2 - 30) * s, (w / 2 + 30) * s, (h / 2 + 30) * s], fill=(205, 244, 210), outline=(50, 140, 70), width=3 * s)
        d.text((w / 2 * s, h / 2 * s + 3 * s), "₹", font=font(LATIN_SYM, 44 * s), fill=(40, 110, 55), anchor="mm")
    return hand_drawn(w, h, f)


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
    lab = text_img(word, 30, fill=(255, 255, 255))
    if lab.width > w - 30:
        lab = lab.resize((w - 30, int(lab.height * (w - 30) / lab.width)))
    im.alpha_composite(lab, ((w - lab.width) // 2, 55 - lab.height // 2 + 0))
    return im


def jar_sprite():
    w, h = 270, 360

    def f(d, s):
        d.rounded_rectangle([14 * s, 70 * s, (w - 14) * s, (h - 6) * s], 46 * s, fill=(210, 240, 255, 70), outline=INK, width=7 * s)
        d.rounded_rectangle([40 * s, 30 * s, (w - 40) * s, 80 * s], 14 * s, fill=(255, 140, 60), outline=INK, width=7 * s)
        d.line([(50 * s, 120 * s), (50 * s, 250 * s)], fill=(255, 255, 255, 190), width=12 * s)
    return hand_drawn(w, h, f)


def piggy_sprite():
    w, h = 520, 400

    def f(d, s):
        P, PD = (255, 158, 196), (226, 95, 150)
        for x in (130, 360):  # legs
            d.rounded_rectangle([x * s, 290 * s, (x + 60) * s, 385 * s], 20 * s, fill=PD, outline=INK, width=7 * s)
        d.ellipse([40 * s, 70 * s, 480 * s, 340 * s], fill=P, outline=INK, width=8 * s)
        d.polygon([(130 * s, 95 * s), (170 * s, 20 * s), (215 * s, 85 * s)], fill=PD, outline=INK, width=7 * s)
        d.ellipse([400 * s, 160 * s, 505 * s, 255 * s], fill=PD, outline=INK, width=7 * s)
        for x in (432, 468):
            d.ellipse([(x - 9) * s, 196 * s, (x + 9) * s, 220 * s], fill=INK)
        d.ellipse([340 * s, 140 * s, 372 * s, 172 * s], fill=INK)
        d.ellipse([350 * s, 146 * s, 360 * s, 156 * s], fill=(255, 255, 255))
        d.rounded_rectangle([200 * s, 80 * s, 330 * s, 98 * s], 9 * s, fill=INK)
        d.arc([10 * s, 160 * s, 80 * s, 230 * s], 80, 330, fill=INK, width=7 * s)
    return hand_drawn(w, h, f)


def board_sprite():
    w, h = 1100, 640

    def f(d, s):
        d.rounded_rectangle([0, 0, w * s, h * s], 28 * s, fill=(168, 104, 52), outline=INK, width=8 * s)
        d.rounded_rectangle([34 * s, 34 * s, (w - 34) * s, (h - 34) * s], 16 * s, fill=(46, 102, 78), outline=INK, width=6 * s)
        d.rounded_rectangle([(w - 260) * s, (h - 60) * s, (w - 120) * s, (h - 44) * s], 6 * s, fill=(250, 250, 240))
    return hand_drawn(w, h, f)


def tick_sprite(r=46):
    def f(d, s):
        R = r * s
        d.ellipse([4 * s, 4 * s, 2 * R - 4 * s, 2 * R - 4 * s], fill=SAVE, outline=INK, width=6 * s)
        d.line([(R * 0.5, R * 1.02), (R * 0.85, R * 1.38), (R * 1.5, R * 0.62)], fill=(255, 255, 255), width=int(R * 0.22), joint="curve")
    return hand_drawn(2 * r, 2 * r, f)


def button_sprite(label):
    lab = text_img(label, 64, fill=(255, 255, 255))
    w, h = lab.width + 120, 150

    def f(d, s):
        d.rounded_rectangle([6 * s, 16 * s, (w - 6) * s, (h - 4) * s], 60 * s, fill=(150, 20, 30), outline=INK, width=7 * s)
        d.rounded_rectangle([6 * s, 6 * s, (w - 6) * s, (h - 18) * s], 60 * s, fill=(235, 40, 50), outline=INK, width=7 * s)
    im = hand_drawn(w, h, f)
    im.alpha_composite(lab, ((w - lab.width) // 2, (h - 18 - lab.height) // 2 + 4))
    return im


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


def mark_sprite(ch, col):
    return text_img(ch, 150, fill=col, stroke=9, stroke_fill=INK)


# ----------------------------------------------------------- backgrounds ---
@lru_cache(maxsize=None)
def room_bg():
    """Bright Indian living room: window, curtains, plant, sofa, tiled floor."""
    def f(d, s):
        S = lambda *v: [x * s for x in v]
        d.rectangle(S(0, 0, W, 720), fill=(255, 226, 170))
        for y in range(40, 720, 90):          # wallpaper dots
            for x in range((y // 90 % 2) * 60, W, 120):
                d.ellipse(S(x - 7, y - 7, x + 7, y + 7), fill=(255, 210, 140))
        d.rectangle(S(0, 700, W, 728), fill=(196, 120, 70), outline=INK, width=5 * s)
        d.rectangle(S(0, 728, W, H), fill=(240, 196, 140))
        for x in range(-600, W + 600, 170):   # floor tiles in perspective
            d.line(S(W / 2 + (x - W / 2) * 0.45, 728, x, H), fill=(214, 164, 108), width=4 * s)
        for y in (790, 880, 1000):
            d.line(S(0, y, W, y), fill=(214, 164, 108), width=4 * s)
        # window
        d.rounded_rectangle(S(110, 110, 560, 520), 24 * s, fill=(255, 255, 255), outline=INK, width=8 * s)
        d.rectangle(S(135, 135, 535, 495), fill=(140, 210, 255))
        d.ellipse(S(410, 160, 500, 250), fill=(255, 220, 60), outline=INK, width=5 * s)
        for cx, cy in ((230, 230), (330, 300)):
            for dx, r in ((-40, 34), (0, 46), (44, 34)):
                d.ellipse(S(cx + dx - r, cy - r, cx + dx + r, cy + r), fill=(255, 255, 255))
        d.rectangle(S(135, 400, 535, 495), fill=(110, 200, 110))
        d.line(S(335, 135, 335, 495), fill=INK, width=7 * s)
        d.line(S(135, 315, 535, 315), fill=INK, width=7 * s)
        for x0, x1 in ((70, 170), (500, 600)):  # curtains
            d.rounded_rectangle(S(x0, 90, x1, 590), 30 * s, fill=NEED, outline=INK, width=6 * s)
        d.rounded_rectangle(S(50, 70, 620, 100), 12 * s, fill=(150, 90, 50), outline=INK, width=6 * s)
        # sofa (right)
        d.rounded_rectangle(S(1180, 470, 1760, 720), 40 * s, fill=(77, 150, 255), outline=INK, width=8 * s)
        d.rounded_rectangle(S(1140, 560, 1240, 760), 30 * s, fill=(60, 120, 220), outline=INK, width=8 * s)
        d.rounded_rectangle(S(1700, 560, 1800, 760), 30 * s, fill=(60, 120, 220), outline=INK, width=8 * s)
        d.rounded_rectangle(S(1230, 600, 1710, 760), 26 * s, fill=(100, 170, 255), outline=INK, width=8 * s)
        d.rounded_rectangle(S(1290, 500, 1420, 600), 26 * s, fill=WANT, outline=INK, width=6 * s)
        # plant (far right)
        for a in range(-60, 61, 30):
            r = math.radians(a - 90)
            d.ellipse(S(1850 + 90 * math.cos(r) - 34, 470 + 110 * math.sin(r) - 70, 1850 + 90 * math.cos(r) + 34, 470 + 110 * math.sin(r) + 70),
                      fill=(80, 190, 90), outline=INK, width=5 * s)
        d.polygon(S(1790, 520, 1910, 520, 1890, 700, 1810, 700), fill=(230, 110, 60), outline=INK, width=6 * s)
    return hand_drawn(W, H, f, ss=1).convert("RGB")


@lru_cache(maxsize=None)
def burst_bg(c1, c2, cx=0.5, cy=0.55, rays=22):
    def f(d, s):
        d.rectangle([0, 0, W * s, H * s], fill=c1)
        ox, oy, R = W * cx * s, H * cy * s, 2600 * s
        for i in range(rays):
            a0 = 2 * math.pi * i / rays
            a1 = a0 + math.pi / rays
            d.polygon([(ox, oy), (ox + R * math.cos(a0), oy + R * math.sin(a0)), (ox + R * math.cos(a1), oy + R * math.sin(a1))], fill=c2)
    return hand_drawn(W, H, f, ss=1).convert("RGB")


@lru_cache(maxsize=None)
def card_bg():
    """Cream title-card with sparkles and confetti (like the reference inserts)."""
    rnd = random.Random(7)

    def f(d, s):
        d.rectangle([0, 0, W * s, H * s], fill=(255, 244, 214))
        cols = [(255, 190, 60), (60, 200, 190), (255, 120, 100), (90, 150, 255)]
        for _ in range(26):
            x, y, r = rnd.randint(60, W - 60), rnd.randint(60, H - 60), rnd.randint(18, 42)
            if 420 < x < 1500 and 300 < y < 780:
                continue
            c = rnd.choice(cols)
            pts = []
            for k in range(8):
                rr = r if k % 2 == 0 else r * 0.35
                a = math.pi / 4 * k
                pts.append(((x + rr * math.cos(a)) * s, (y + rr * math.sin(a)) * s))
            d.polygon(pts, fill=c)
        for _ in range(30):
            x, y = rnd.randint(30, W - 30), rnd.randint(30, H - 30)
            if 420 < x < 1500 and 300 < y < 780:
                continue
            d.ellipse([(x - 8) * s, (y - 8) * s, (x + 8) * s, (y + 8) * s], fill=rnd.choice(cols))
    return hand_drawn(W, H, f, ss=1).convert("RGB")


# ------------------------------------------------------------- character ---
MOUTHS = {"happy": (92, 220, 175, 268), "stand": (84, 176, 148, 208)}


@lru_cache(maxsize=None)
def boy(pose, mouth="open"):
    """Boy sprite. mouth: open (as drawn) | mid | closed (drawn smile line)."""
    im = Image.open(SPRITES / f"{pose}.png").convert("RGBA")
    if mouth == "open" or pose not in MOUTHS:
        return im
    x0, y0, x1, y1 = MOUTHS[pose]
    mouth_px = im.crop((x0, y0, x1, y1))
    skin = im.getpixel((x0 - 6, (y0 + y1) // 2))[:3]
    if sum(skin) < 300:
        skin = SKIN
    out = im.copy()
    d = ImageDraw.Draw(out)
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


def draw_boy(frame, pose, cx, foot_y, scale, local, talking=False, hop=0.0, bob=True):
    mouth = talk_state(local, talking)
    sp = boy(pose, mouth)
    breathe = 1 + 0.012 * math.sin(local * 2 * math.pi / 1.4) if bob else 1
    tilt = 1.6 * math.sin(local * 2 * math.pi / 2.3) if talking else 0
    s = scale
    sp2 = sp.resize((int(sp.width * s), int(sp.height * s * breathe)), Image.BILINEAR)
    place(frame, sp2, cx, foot_y - hop, 1.0, rot=tilt, anchor="bottom")


# --------------------------------------------------------------- camera ----
def camera(img, zoom=1.0, fx=0.5, fy=0.5, shake=0.0, local=0.0):
    if zoom <= 1.001 and shake == 0:
        return img
    cw, ch = W / zoom, H / zoom
    x = clamp(fx * W - cw / 2, 0, W - cw) + shake * math.sin(local * 55)
    y = clamp(fy * H - ch / 2, 0, H - ch) + shake * math.cos(local * 47)
    x, y = clamp(x, 0, W - cw), clamp(y, 0, H - ch)
    return img.resize((W, H), Image.BILINEAR, box=(x, y, x + cw, y + ch))


# ----------------------------------------------------------------- shots ---
class Shot:
    def __init__(self, dur, fn, cut_sfx="whoosh", events=()):
        self.dur, self.fn, self.cut_sfx, self.events = dur, fn, cut_sfx, list(events)


SFX = []          # (time, kind)


def sfx_at(time, kind):
    SFX.append((round(time, 3), kind))


def title_card(text, palette=((255, 120, 90), (60, 190, 180), (90, 140, 255))):
    """Big bouncy multi-colour words on the confetti card."""
    words = text.split()
    size = 170
    while size > 70:
        lines = wrap(text, size, W - 360)
        if len(lines) <= 3:
            break
        size -= 10
    lines = wrap(text, size, W - 360)
    imgs = [text_img(l, size, fill=palette[i % len(palette)], stroke=12, stroke_fill=(255, 255, 255), shadow=10) for i, l in enumerate(lines)]

    def fn(local, dur, g0):
        fr = card_bg().copy()
        total = sum(i.height for i in imgs) + 20 * (len(imgs) - 1)
        y = H / 2 - total / 2
        for k, im in enumerate(imgs):
            sc = pop(local, 0.08 + 0.12 * k, 0.4)
            wob = 2.0 * math.sin(local * 6 + k)
            place(fr, im, W / 2, y + im.height / 2, sc, rot=wob)
            y += im.height + 20
        return camera(fr, 1.0 + 0.04 * local / dur, 0.5, 0.5)
    return fn


def make_shots(seg, T, g0):
    """Return a list of Shot for one segment, scaled to fit T seconds."""
    v = seg["visual"]
    shots = []

    def S(d, fn, sfx="whoosh", events=()):
        shots.append(Shot(d, fn, sfx, events))

    if v == "title":
        heading = caps(t(seg["heading"]))
        cal = seg.get("calendar")
        cal_word = t(seg.get("calendar_label", {"en": "SALARY", "ta": "சம்பளம்", "hi": "SALARY"}))
        note = note_sprite()
        S(1.4, title_card(heading), "pop")

        def walk(local, dur, _):
            fr = room_bg().copy()
            if cal:
                place(fr, calendar_sprite(cal[0], cal_word), 1420, 330, 1.2 * pop(local, 0.5, 0.35))
            x = -200 + (820 + 200) * ease_out(local / 1.1)
            hop = abs(math.sin(local * 9)) * 22 if local < 1.1 else 0
            draw_boy(fr, "stand", x, 1030, 1.55, local, talking=local > 0.9, hop=hop)
            place(fr, note, x + 200, 680 - hop + 6 * math.sin(local * 5), 1.1, rot=12)
            place(fr, note, x + 215, 645 - hop + 6 * math.sin(local * 5 + 1), 1.1, rot=-6)
            return camera(fr, 1.22 + 0.04 * local / dur, 0.55, 0.55)
        S(1.6, walk)
        if cal:
            def flip(local, dur, _):
                fr = room_bg().copy()
                # flip page: squash old to 0, grow new
                ft = 0.35
                if local < ft:
                    place(fr, calendar_sprite(cal[0], cal_word).resize((250, max(2, int(280 * (1 - local / ft))))), 1500, 300)
                else:
                    sc = back((local - ft) / 0.35)
                    place(fr, calendar_sprite(cal[1], cal_word).resize((250, max(2, int(280 * clamp(sc, 0.01, 1.3))))), 1500, 300)
                draw_boy(fr, "turn", 960, 1030, 1.55, local, talking=True)
                for k in range(3):     # cash flies out of the window
                    st = 0.5 + 0.22 * k
                    p = clamp((local - st) / 0.8)
                    if p <= 0:
                        place(fr, note, 1070, 690 - 30 * k, 1.0, rot=10 - 8 * k)
                    elif p < 1:
                        x = 1070 + (330 - 1070) * p
                        y = 690 - 30 * k - math.sin(p * math.pi) * 380 - 250 * p
                        place(fr, note, x, y, 1 - 0.6 * p, rot=200 * p)
                return camera(fr, 1.35, 0.62, 0.45)
            S(1.6, flip, events=[(0.35, "flip")] + [(0.5 + 0.22 * k, "whoosh") for k in range(3)])
        mood = seg.get("mood", "sad")

        def react(local, dur, _):
            fr = burst_bg((120, 170, 255), (150, 195, 255)).copy()
            draw_boy(fr, mood, W / 2, H + 40, 2.75, local, talking=False)
            place(fr, mark_sprite("?", (255, 255, 255)), W / 2 + 470, 330, pop(local, 0.15))
            place(fr, mark_sprite("?", WANT), W / 2 + 600, 470, pop(local, 0.3), rot=-15)
            return camera(fr, 1.0 + 0.05 * local / dur, 0.5, 0.45, shake=3 if local < 0.3 else 0, local=local)
        S(max(1.0, T - sum(s.dur for s in shots)), react)

    elif v == "split":
        parts = seg["parts"]
        labels = t(seg["labels"])
        cols = [NEED, WANT, SAVE, POP]
        S(1.3, title_card("-".join(str(p) for p in parts)), "pop")
        jar = jar_sprite()
        coin = coin_sprite(40)
        jx = [920, 1305, 1690][:len(parts)]
        n_coins = [max(1, round(p / 10)) for p in parts]
        order = [(j, k) for j, n in enumerate(n_coins) for k in range(n)]
        tags = [text_img(lab, 64, fill=cols[j % 4], stroke=7, stroke_fill=INK) for j, lab in enumerate(labels)]
        tags = [tg if tg.width <= 350 else tg.resize((350, int(tg.height * 350 / tg.width)), Image.LANCZOS) for tg in tags]
        pct_big = [text_img(f"{p}%", 130, fill=cols[j % 4], stroke=12, stroke_fill=INK, shadow=8) for j, p in enumerate(parts)]
        pcts = [text_img(f"{p}%", 76, fill=(255, 255, 255), stroke=8, stroke_fill=INK) for p in parts]
        fly = 0.32

        def jars(local, dur, _, zoomed=False):
            fr = room_bg().copy()
            draw_boy(fr, "stand", 560, 1030, 1.5, local, talking=not zoomed)
            filled = [0] * len(parts)
            for idx, (j, k) in enumerate(order):
                st = 0.6 + idx * fly
                p = (local - st) / 0.45
                if zoomed or p >= 1:
                    filled[j] += 1
                elif p > 0:
                    x = 600 + (jx[j] - 600) * p
                    y = 640 - math.sin(p * math.pi) * 330 + (560 - 640) * p
                    place(fr, coin, x, y, 1.0, rot=360 * p)
            d = ImageDraw.Draw(fr)
            for j, x in enumerate(jx):
                level = filled[j] / n_coins[j] * parts[j] / max(parts)
                top = 980 - 6 - int(250 * level)
                if level > 0:
                    d.rounded_rectangle([x - 115, top, x + 115, 968], 36, fill=cols[j % 4])
                place(fr, jar, x, 985, 1.0, anchor="bottom")
                place(fr, tags[j], x, 575, 1.0 if zoomed else pop(local, 0.15 + 0.12 * j))
                done_t = 0.6 + (sum(n_coins[:j + 1]) - 1) * fly + 0.45
                if zoomed or local > done_t:
                    sc = 1.0 + (0.06 * math.sin(local * 6 + j) if zoomed else 0)
                    place(fr, pcts[j], x, 800, sc if zoomed else pop(local, done_t))
            return camera(fr, 1.18 + 0.03 * local / dur, 0.62, 0.62)
        coin_time = 0.6 + len(order) * fly + 0.5
        S(max(3.0, coin_time + 0.4), jars, events=[(0.6 + i * fly + 0.45, "coin") for i in range(len(order))])

        def jars_close(local, dur, g):
            fr = jars(local, dur, g, zoomed=True)
            return camera(fr, 1.25 + 0.04 * local / dur, 0.72, 0.65)

        def think(local, dur, _):
            fr = burst_bg((255, 200, 80), (255, 220, 120)).copy()
            draw_boy(fr, "think", W / 2 - 250, H + 40, 2.75, local)
            for j in range(len(parts)):
                place(fr, pct_big[j], 1250 + 230 * j, 300 + 170 * j, pop(local, 0.15 + 0.15 * j), rot=-8)
            return camera(fr, 1.0 + 0.05 * local / dur, 0.5, 0.45)
        rest = T - sum(s.dur for s in shots)
        if rest > 3.0:
            S(rest - 1.6, jars_close)
            S(1.6, think)
        else:
            S(max(1.0, rest), think)

    elif v == "number":
        value = str(seg["value"])
        digits = int("".join(ch for ch in value if ch.isdigit()) or 0)
        prefix = value[: next((i for i, ch in enumerate(value) if ch.isdigit()), 0)]
        sub = t(seg.get("caption", ""))
        piggy, coin = piggy_sprite(), coin_sprite(40)
        sub_img = text_block(sub, 900, 64, 40, 2, fill=(255, 255, 255), stroke=7, stroke_fill=INK) if sub else None

        def wow(local, dur, _):
            fr = burst_bg((255, 170, 60), (255, 200, 90)).copy()
            draw_boy(fr, "wow", W / 2, H + 40, 2.75, local)
            place(fr, mark_sprite("!", NEED), W / 2 + 470, 330, pop(local, 0.1), rot=10)
            place(fr, mark_sprite("!", WANT), W / 2 - 480, 360, pop(local, 0.25), rot=-12)
            return camera(fr, 1.0 + 0.06 * local / dur, 0.5, 0.45, shake=4 if local < 0.35 else 0, local=local)
        S(1.3, wow, "pop")

        def count(local, dur, _):
            fr = burst_bg((100, 200, 120), (130, 220, 150)).copy()
            place(fr, piggy, W / 2, 930, 1.05 + 0.03 * math.sin(local * 8), anchor="bottom")
            for k in range(9):
                st = 0.1 + k * 0.22
                p = clamp((local - st) / 0.5)
                if 0 < p < 1:
                    place(fr, coin, W / 2 - 30 + (k % 3 - 1) * 60 * (1 - p), -60 + 560 * p, 1.0, rot=300 * p)
            n = int(digits * ease_out(local / 2.2))
            s_ = f"{n:,}" if TEXT_LANG == "en" or True else str(n)
            if digits >= 100000:   # Indian grouping for lakhs
                s_ = f"{n // 100000},{n // 1000 % 100:02d},{n % 1000:03d}" if n >= 100000 else f"{n:,}"
            num = text_img(prefix + s_, 190, fill=(255, 255, 255), stroke=14, stroke_fill=INK, shadow=10)
            place(fr, num, W / 2, 210, pop(local, 0.0, 0.3) * (1 + 0.05 * math.sin(local * 7)))
            if sub_img is not None:
                place(fr, sub_img, W / 2, 390, pop(local, 0.4))
            return camera(fr, 1.0 + 0.03 * local / dur, 0.5, 0.5)
        S(2.8, count, events=[(0.1 + k * 0.22 + 0.45, "coin") for k in range(9)])
        badge = text_img(value, 150, fill=SAVE, stroke=12, stroke_fill=INK, shadow=8)

        def happy(local, dur, _):
            fr = burst_bg((255, 210, 80), (255, 230, 130)).copy()
            draw_boy(fr, "happy", W / 2 - 300, H + 40, 2.75, local, talking=True)
            place(fr, badge, 1380, 470, pop(local, 0.2) * (1 + 0.04 * math.sin(local * 6)), rot=-6)
            return camera(fr, 1.0 + 0.05 * local / dur, 0.5, 0.45)
        S(max(1.0, T - 4.1), happy)

    elif v == "bullets":
        items = t(seg["items"])
        board, tick = board_sprite(), tick_sprite(42)
        rows = [text_block(it, 860, 74, 44, 1, fill=(250, 250, 240), stroke=0) for it in items]
        n = len(items)
        per = T / n

        def board_shot(k):
            def fn(local, dur, _):
                fr = room_bg().copy()
                place(fr, board, 1240, 40 + 320, 1.0)
                y0 = 360 - (n - 1) * 75
                for i in range(k + 1):
                    a = 1.0 if i < k else clamp(local / 0.25)
                    place(fr, rows[i], 860 + rows[i].width / 2, y0 + i * 150, 1.0, alpha=a)
                    place(fr, tick, 780, y0 + i * 150, 1.0 if i < k else pop(local, 0.35))
                draw_boy(fr, "turn", 430, 1030, 1.5, local, talking=True, hop=abs(math.sin(local * 8)) * 18 if local < 0.4 else 0)
                return camera(fr, 1.15 + 0.04 * local / dur, 0.52, 0.45)
            return fn

        def nod(local, dur, _):
            fr = burst_bg((120, 210, 200), (150, 228, 218)).copy()
            draw_boy(fr, "happy", W / 2, H + 40, 2.75, local, talking=True)
            return camera(fr, 1.0 + 0.05 * local / dur, 0.5, 0.45)
        for k in range(n):
            if per > 2.4 and k < n - 1:
                S(per - 0.9, board_shot(k), "pop" if k else "whoosh")
                S(0.9, nod)
            else:
                S(per, board_shot(k), "pop")

    elif v == "outro":
        note = caps(t(seg["heading"]))
        btn = button_sprite("SUBSCRIBE")
        btn_done = button_sprite("SUBSCRIBED ✓" if False else "SUBSCRIBED")
        bell = bell_sprite()
        note_img = text_block(note, 1000, 54, 34, 2, fill=(255, 255, 255), stroke=6, stroke_fill=INK)

        def sub(local, dur, _):
            fr = burst_bg((210, 90, 160), (230, 120, 185)).copy()
            hop = abs(math.sin(local * 5)) * 60 if local < 1.3 else 0
            draw_boy(fr, "stand", 470, 1040, 1.6, local, talking=local > 1.3, hop=hop)
            press = 1.7
            b = btn if local < press else btn_done
            sc = pop(local, 0.4, 0.4)
            if press - 0.08 < local < press + 0.12:
                sc *= 0.9
            place(fr, b, 1180, 420, 1.5 * sc)
            place(fr, bell, 1180 + btn.width * 0.75 + 140, 420, 1.3 * pop(local, 0.9), rot=18 * math.sin(local * 18) * math.exp(-(local - 2.0) * 2) if local > 2.0 else 0)
            place(fr, note_img, 1250, 720, pop(local, 0.6))
            return camera(fr, 1.0 + 0.03 * local / dur, 0.5, 0.5)
        S(T, sub, "pop", events=[(1.7, "click"), (2.0, "ding")])

    else:
        raise SystemExit(f"unknown visual {v}")

    # fit to T exactly: stretch/squeeze shot lengths proportionally
    tot = sum(s.dur for s in shots)
    for s in shots:
        s.dur *= T / tot
    return shots


# -------------------------------------------------------------- captions ---
def caption_chunks(seg, start, T):
    """Short all-caps chunks spread across the segment by text length."""
    if not SCRIPT.get("captions", True) or "say" not in seg:
        return []
    text = caps(t(seg["say"]) if LANG != "hi" else (seg["say"].get("hinglish") or seg["say"].get("en")))
    words = text.split()
    per = 3 if TEXT_LANG == "en" else 2
    chunks, cur = [], []
    for w_ in words:
        cur.append(w_)
        if len(cur) >= per or w_[-1:] in ".?!,:":
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    v = VOICE.get(seg["id"])
    if v and v["words"]:
        out, cur, st = [], [], None
        for a, b, w_ in v["words"]:
            if not cur:
                st = a
            cur.append(w_)
            if len(cur) >= per or w_[-1:] in ".?!,:;":
                out.append([start + VOICE_LEAD + st, start + VOICE_LEAD + b, caps(" ".join(cur))])
                cur = []
        if cur:
            out.append([start + VOICE_LEAD + st, start + VOICE_LEAD + v["words"][-1][1], caps(" ".join(cur))])
        for k in range(len(out) - 1):          # hold each chunk until the next one
            out[k][1] = out[k + 1][0]
        out[-1][1] = min(start + T, out[-1][1] + 0.5)
        return [tuple(c) for c in out]
    lead, tail = 0.25, 0.35
    span = max(0.5, T - lead - tail)
    total = sum(len(c) for c in chunks)
    out, t0 = [], start + lead
    for c in chunks:
        d = span * len(c) / total
        out.append((t0, t0 + d, c))
        t0 += d
    return out


@lru_cache(maxsize=256)
def caption_img(text, size):
    return text_block(text, 1500 if size > 70 else 980, size, 40, 2, fill=(255, 255, 255), stroke=9, stroke_fill=(0, 0, 0))


# ---------------------------------------------------------------- audio ----
def synth_sfx(length, path, rate=44100):
    n = int((length + 0.5) * rate)
    track = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(3)

    def tone(dur, f0, f1, vol, kind="sine"):
        tt = np.arange(int(dur * rate)) / rate
        f = np.linspace(f0, f1, len(tt))
        ph = 2 * np.pi * np.cumsum(f) / rate
        env = np.exp(-tt / (dur * 0.35))
        return (np.sin(ph) * env * vol).astype(np.float32)

    def whoosh(dur=0.35, vol=0.18):
        m = int(dur * rate)
        noise = rng.standard_normal(m).astype(np.float32)
        k = 40
        noise = np.convolve(noise, np.ones(k) / k, mode="same")
        env = np.sin(np.linspace(0, np.pi, m)) ** 2
        return noise * env * vol * 3

    bank = {
        "pop": lambda: tone(0.12, 500, 1300, 0.30),
        "coin": lambda: np.concatenate([tone(0.08, 1800, 1800, 0.22), tone(0.25, 2400, 2400, 0.22)]),
        "whoosh": whoosh,
        "flip": lambda: whoosh(0.18, 0.25),
        "click": lambda: tone(0.06, 900, 400, 0.35),
        "ding": lambda: tone(0.6, 1320, 1320, 0.25) + np.pad(tone(0.6, 1980, 1980, 0.08)[: int(0.6 * rate)], (0, 0)),
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
    head = t(first["heading"]) if first else SCRIPT.get("title", "")
    return caps(f"EP {SCRIPT.get('ep', 0)}: {head}")


def logo_img():
    c = coin_sprite(70)
    return c


def main():
    out_dir = Path("out/media/cartoon")
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline, caps_list, g = [], [], 0.0
    for seg in SCRIPT["segments"]:
        T = seg_time(seg)
        SEG_START[seg["id"]] = g
        for sh in make_shots(seg, T, g):
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

    # Short layout pieces
    title = text_block(series_title(), 1000, 64, 36, 2, fill=YELLOW_T, stroke=0)
    logo = logo_img()

    def ff(path, w, h):
        return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                 "-crf", "20", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    long_raw, short_raw = out_dir / f"{LANG}_v.mp4", out_dir / f"{LANG}_short_v.mp4"
    p_long, p_short = ff(long_raw, W, H), ff(short_raw, SW, SH)

    si = 0
    for fi in range(nframes):
        tm = fi / FPS
        while si + 1 < len(timeline) and timeline[si + 1][0] <= tm:
            si += 1
        g0, sh = timeline[si]
        local = tm - g0
        NOW[0] = tm
        panel = sh.fn(local, sh.dur, g0)
        if local < 0.12:                 # punch-in on every cut
            panel = camera(panel, 1.0 + 0.05 * (1 - local / 0.12))
        cap = next((c for c in caps_list if c[0] <= tm < c[1]), None)

        long_f = panel.copy()
        if cap:
            ci = caption_img(cap[2], 84)
            sc = pop(tm - cap[0], 0.0, 0.18)
            place(long_f, ci, W / 2, H - 40 - ci.height / 2, 0.85 + 0.15 * sc)
        p_long.stdin.write(long_f.tobytes())

        short_f = Image.new("RGB", (SW, SH), (0, 0, 0))
        short_f.paste(panel.resize((SW, int(SW * H / W)), Image.BILINEAR), (0, PANEL_Y))
        place(short_f, title, SW / 2, PANEL_Y - 30 - title.height / 2)
        place(short_f, logo, SW - 110, 150, 1.0)
        if cap:
            ci = caption_img(cap[2], 64)
            place(short_f, ci, SW / 2, PANEL_Y + 608 + 50 + ci.height / 2, 0.85 + 0.15 * pop(tm - cap[0], 0.0, 0.18))
        p_short.stdin.write(short_f.tobytes())
        if fi % 150 == 0:
            print(f"frame {fi}/{nframes}", flush=True)
    for p in (p_long, p_short):
        p.stdin.close()
        p.wait()

    wav = out_dir / f"{LANG}_sfx.wav"
    synth_sfx(length, wav)
    for raw, final in ((long_raw, out_dir / f"{LANG}.mp4"), (short_raw, out_dir / f"{LANG}_short.mp4")):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", str(wav), "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)], check=True)
        raw.unlink()
    print("wrote", out_dir / f"{LANG}.mp4", out_dir / f"{LANG}_short.mp4", f"{length:.1f}s")


if __name__ == "__main__":
    main()
