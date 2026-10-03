"""Puppet show renderer (v3): the family characters acting out a short scene.

    python pipeline/puppet_show.py shows/ep01-three-jars            # full 9:16 video
    python pipeline/puppet_show.py shows/ep01-three-jars --stills 1.2,5,9.5
    python pipeline/puppet_show.py shows/ep01-three-jars --range 0 6

Inputs: shows/<ep>/show.yaml, the cut sprites in assets/family/<char>/cut/,
and (optional) out/show/voice.json + out/show/<beat>.wav from tts_show.py.
Without voice files it estimates the timing and renders a silent draft.

What makes it move like an animated show (all free, CPU only):
  * a puppet rig: each full-body sprite bends at the neck (head nods, tilts and
    sways while talking) and leans from the feet, so the body is never frozen,
  * squash and stretch: anticipation before a jump, stretch on take-off, a
    parabola in the air, a springy squash on landing, pose "pops" with overshoot,
  * talking close-ups: mouth shapes painted on the face, opened by the voice
    loudness, plus blinking eyelids every few seconds,
  * real 2D physics for the coins (gravity, bounce, friction, stacking),
  * a painted room with depth of field, window light, cast and contact
    shadows, a camera that pushes in and punches in on every cut,
  * word-by-word karaoke captions and light sound effects.
"""
import json
import math
import os
import random
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 720, 1280, 24
AUDIO_RATE = 24000
FONT = str(ROOT / "assets" / "fonts" / "Poppins-Bold.ttf")
WORLD_W, WORLD_H = 900, 1460          # the painted room (world units = pixels at zoom 1)
CAM0 = (450.0, 770.0)                 # default camera centre in the room
FLOOR_Y = 1010                        # where the wall meets the floor (world y)
FEET_Y = 1312                         # where the characters stand (world y)

CHARS = {
    # on-screen standing height (px), neck position as a fraction of the sprite height
    "dad":   {"h": 690, "neck": 0.205},
    "aarav": {"h": 440, "neck": 0.29},
}
# which way each sprite faces as drawn on the sheet (+1 = towards screen right)
FACING = {
    "dad":   {"turn_3q": -1, "turn_3q_r": 1, "pose_point": 1, "pose_wave": -1, "pose_jump": 1,
              "pose_run": 1, "pose_walk": -1, "turn_front": 1},
    "aarav": {"turn_3q": 1, "pose_point": -1, "pose_wave": 1, "pose_jump": 1, "pose_run": 1,
              "pose_walk": -1, "turn_front": 1},
}
# face landmarks on the expression sprites (pixels in the cut PNG):
# eyes (x, y) and half-width; mouth centre (x, y) and width
FACES = {
    ("dad", "ex_happy"):     {"eyes": [(59, 70), (93, 70)], "erx": 9.5, "ery": 7.0, "mouth": (78, 95.5), "mw": 33},
    ("dad", "ex_calm"):      {"eyes": [(69, 56), (95, 55)], "erx": 9.0, "ery": 7.0, "mouth": (85, 81.5), "mw": 29},
    ("aarav", "ex_happy"):   {"eyes": [(36, 74), (73, 74)], "erx": 8.5, "ery": 7.5, "mouth": (55, 99.5), "mw": 30},
    ("aarav", "ex_curious"): {"eyes": [(38.5, 73), (77, 72)], "erx": 8.0, "ery": 6.5, "mouth": (62, 97), "mw": 23},
}

# ------------------------------------------------------------------ maths


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def back_out(x, k=1.8):
    x = clamp(x)
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def spring(x, amp=1.0, freq=2.6, decay=6.0):
    """Damped wobble from `amp` to 0 (squash that settles)."""
    if x < 0:
        return 0.0
    return amp * math.exp(-decay * x) * math.cos(2 * math.pi * freq * x)


def noise1(t, seed=0.0):
    """Smooth pseudo-random wobble in [-1, 1]."""
    return (math.sin(1.7 * t + seed) * 0.5 + math.sin(2.9 * t + 1.3 * seed) * 0.3
            + math.sin(4.3 * t + 2.1 * seed) * 0.2)

# ------------------------------------------------------------------ drawing helpers


def premul(rgba_u8):
    a = rgba_u8.astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


def affine(anchor, pos, sx=1.0, sy=1.0, rot=0.0):
    """2x3 matrix: sprite point -> screen, scaling/rotating about `anchor`, which lands on `pos`."""
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    m = np.array([[c * sx, -s * sy, 0.0], [s * sx, c * sy, 0.0]], np.float32)
    m[:, 2] = np.array(pos, np.float32) - m[:, :2] @ np.array(anchor, np.float32)
    return m


def _bbox(m, w, h, pad=0):
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ m.T
    x0 = max(0, int(math.floor(pts[:, 0].min())) - 1 - pad)
    x1 = min(W, int(math.ceil(pts[:, 0].max())) + 1 + pad)
    y0 = max(0, int(math.floor(pts[:, 1].min())) - 1 - pad)
    y1 = min(H, int(math.ceil(pts[:, 1].max())) + 1 + pad)
    return x0, y0, x1, y1


def blit(canvas, px, m, alpha=1.0):
    """Composite premultiplied RGBA `px` onto RGB `canvas` through affine `m`."""
    h, w = px.shape[:2]
    x0, y0, x1, y1 = _bbox(m, w, h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0.002:
        return
    m2 = m.copy()
    m2[:, 2] -= (x0, y0)
    out = cv2.warpAffine(px, m2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if alpha < 1:
        out *= alpha
    reg = canvas[y0:y1, x0:x1]
    reg[:] = out[..., :3] + reg * (1 - out[..., 3:4])


def blit_shadow(canvas, alpha_img, m, strength=0.4, blur=6.0, tint=(0.75, 0.68, 0.72)):
    """Darken the canvas with a warped, blurred silhouette."""
    h, w = alpha_img.shape[:2]
    pad = int(blur * 3)
    x0, y0, x1, y1 = _bbox(m, w, h, pad)
    if x1 <= x0 or y1 <= y0:
        return
    m2 = m.copy()
    m2[:, 2] -= (x0, y0)
    a = cv2.warpAffine(alpha_img, m2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if blur > 0:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    a = (a * strength)[..., None]
    reg = canvas[y0:y1, x0:x1]
    reg[:] = reg * (1 - a) + reg * a * np.array(tint, np.float32) * 0.45


def ellipse_shadow(canvas, cx, cy, rx, ry, strength=0.45, blur=8):
    x0, x1 = int(max(0, cx - rx - 3 * blur)), int(min(W, cx + rx + 3 * blur))
    y0, y1 = int(max(0, cy - ry - 3 * blur)), int(min(H, cy + ry + 3 * blur))
    if x1 <= x0 or y1 <= y0:
        return
    lay = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.ellipse(lay, (int(cx - x0), int(cy - y0)), (max(1, int(rx)), max(1, int(ry))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), blur) * strength
    canvas[y0:y1, x0:x1] *= (1 - lay[..., None] * 0.85)


def text_image(text, size, fill=(255, 255, 255), stroke=0, stroke_fill=(0, 0, 0), font=FONT):
    f = ImageFont.truetype(font, size)
    l, t, r, b = f.getbbox(text, stroke_width=stroke)
    im = Image.new("RGBA", (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((2 - l, 2 - t), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return premul(np.asarray(im))

# ------------------------------------------------------------------ sprites and the puppet rig


def unsharp(px, amount=0.45, sigma=1.1):
    rgb = px[..., :3]
    blur = cv2.GaussianBlur(rgb, (0, 0), sigma)
    px[..., :3] = np.clip(rgb + amount * (rgb - blur), 0, None)
    px[..., :3] = np.minimum(px[..., :3], px[..., 3:4])
    return px


def room_grade(px):
    """Warm room light, window rim on the left edge, soft shade on the right."""
    a = px[..., 3]
    h, w = a.shape
    px[..., :3] *= np.array([1.0, 0.965, 0.92], np.float32)
    shade = np.linspace(1.04, 0.9, w, dtype=np.float32)[None, :, None]
    px[..., :3] *= shade
    er = cv2.erode(a, np.ones((5, 5), np.uint8))
    edge = np.clip(a - er, 0, 1)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    left = np.clip(gx, 0, None)                          # alpha rising to the right = a left edge
    left = left / (left.max() + 1e-6)
    rim = np.clip(edge * left * 1.6, 0, 1)[..., None]
    px[..., :3] += rim * np.array([1.0, 0.9, 0.7], np.float32) * 0.35 * px[..., 3:4]
    px[..., :3] = np.minimum(px[..., :3], px[..., 3:4])
    return px


class Cast:
    """Loads cut sprites at screen scale (cached)."""

    def __init__(self):
        self.cache = {}
        self.raw = {}

    def path(self, char, name):
        return ROOT / "assets" / "family" / char / "cut" / f"{name}.png"

    def size(self, char, name):
        if (char, name) not in self.raw:
            self.raw[(char, name)] = Image.open(self.path(char, name)).convert("RGBA")
        return self.raw[(char, name)].size

    def body_scale(self, char, name):
        ref = "turn_front" if name.startswith("turn_") else "pose_wave"
        return CHARS[char]["h"] / self.size(char, ref)[1]

    def get(self, char, name, scale=None, grade=True):
        scale = scale or self.body_scale(char, name)
        key = (char, name, round(scale, 4), grade)
        if key not in self.cache:
            self.size(char, name)
            im = self.raw[(char, name)]
            w, h = im.size
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            px = premul(np.asarray(im))
            if scale > 1.2:
                px = unsharp(px)
            if grade:
                px = room_grade(px)
            self.cache[key] = np.ascontiguousarray(px)
        return self.cache[key]


_GRIDS = {}


def rig(px, neck, hrot=0.0, hdx=0.0, hdy=0.0, lean=0.0, pad=30):
    """Bend a full-body sprite: the head turns about the neck, the body leans from the feet.

    Returns (warped sprite, anchor at the feet)."""
    h, w = px.shape[:2]
    P = cv2.copyMakeBorder(px, pad, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    hh, ww = P.shape[:2]
    key = (hh, ww)
    if key not in _GRIDS:
        ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
        _GRIDS[key] = (xs, ys)
    xs, ys = _GRIDS[key]
    v = (ys - pad) / h                                        # 0 = top of the sprite, 1 = feet
    wh = np.clip((neck + 0.05 - v) / 0.09, 0, 1)              # 1 on the head, fading over the neck
    wh = wh * wh * (3 - 2 * wh)
    cx, ny = ww / 2.0, pad + neck * h
    th = math.radians(hrot) * wh
    lean_off = lean * np.clip(1 - v, 0, 1) ** 1.4
    dx = xs - cx - hdx * wh - lean_off
    dy = ys - ny - hdy * wh
    c, s = np.cos(th), np.sin(th)
    sx = c * dx + s * dy + cx
    sy = -s * dx + c * dy + ny
    out = cv2.remap(P, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out, (ww / 2.0, float(hh))

# ------------------------------------------------------------------ faces: mouth and blinks


def paint_mouth(px, cx, cy, mw, open_, wide):
    """Paint an open mouth over the closed smile (open_ 0..1)."""
    if open_ < 0.07:
        return
    rx = mw * 0.5 * (0.40 + 0.22 * wide + 0.10 * open_)
    ry = mw * 0.5 * (0.10 + 0.36 * open_)
    ccx, ccy = cx, cy + ry * 0.38
    r = int(max(rx, ry) + 8)
    x0, y0 = int(ccx - r), int(ccy - r)
    x1, y1 = x0 + 2 * r, y0 + 2 * r
    if x0 < 0 or y0 < 0 or x1 > px.shape[1] or y1 > px.shape[0]:
        return
    S = 4                                                         # supersample for clean edges
    def ell(rx_, ry_, ox=0.0, oy=0.0):
        m = np.zeros((2 * r * S, 2 * r * S), np.float32)
        cv2.ellipse(m, (int((ccx - x0 + ox) * S), int((ccy - y0 + oy) * S)),
                    (max(1, int(rx_ * S)), max(1, int(ry_ * S))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        return cv2.resize(m, (2 * r, 2 * r), interpolation=cv2.INTER_AREA)
    yy = np.arange(2 * r, dtype=np.float32)[:, None]
    top = np.clip((yy - (ccy - y0 - ry * 0.5)) / 1.5 + 0.5, 0, 1)        # flat upper lip: a "D" shape
    soft = lambda m: cv2.GaussianBlur(m, (0, 0), 0.9)
    lip = soft(ell(rx + 2.0, ry + 2.0, 0, 0.5) * np.clip((yy - (ccy - y0 - ry * 0.5 - 2.0)) / 1.5 + 0.5, 0, 1)) * 0.75
    inner = soft(ell(rx, ry) * top)
    tongue = ell(rx * 0.6, ry * 0.45, 0, ry * 0.6) * inner
    teeth = ell(rx * 0.8, ry * 0.3, 0, -ry * 0.35) * inner * top * clamp((open_ - 0.55) * 3) * 0.7
    reg = px[y0:y1, x0:x1]
    a = reg[..., 3:4]
    def put(mask, col):
        m = mask[..., None]
        reg[..., :3] = reg[..., :3] * (1 - m) + np.array(col, np.float32) * a * m
    put(lip, (0.62, 0.33, 0.30))
    put(inner, (0.30, 0.10, 0.10))
    put(tongue, (0.66, 0.30, 0.31))
    put(teeth, (0.92, 0.88, 0.84))


def make_closed_eyes(px, eyes, rx, ry):
    """A copy of the bust with both eyes shut: the eye area is inpainted from the skin around it
    and a lash line is drawn where the lids meet."""
    out = px.copy()
    a = np.clip(px[..., 3:4], 1e-3, None)
    rgb = np.clip(px[..., :3] / a, 0, 1)
    for ex, ey in eyes:
        R = int(max(rx, ry) * 2.6)
        x0, y0 = int(ex - R), int(ey - R)
        reg = (rgb[y0:y0 + 2 * R, x0:x0 + 2 * R] * 255).astype(np.uint8)
        mask = np.zeros(reg.shape[:2], np.uint8)
        cv2.ellipse(mask, (int(ex - x0), int(ey - y0)), (int(rx * 1.25), int(ry * 1.45)), 0, 0, 360, 255, -1)
        fill = cv2.inpaint(np.ascontiguousarray(reg), mask, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255
        fill = cv2.GaussianBlur(fill, (0, 0), 2.0) * (mask[..., None] / 255.0) + reg.astype(np.float32) / 255 * (1 - mask[..., None] / 255.0)
        # crease shadow above, lash line where the lids meet
        sh = np.zeros(mask.shape, np.float32)
        cv2.ellipse(sh, (int(ex - x0), int(ey - y0 - ry * 0.2)), (int(rx * 1.05), int(ry * 0.9)), 0, 200, 340, 1.0,
                    max(2, int(rx * 0.12)), cv2.LINE_AA)
        sh = cv2.GaussianBlur(sh, (0, 0), rx * 0.08 + 1)
        fill *= (1 - 0.12 * sh[..., None])
        lash = np.zeros(mask.shape, np.float32)
        cv2.ellipse(lash, (int(ex - x0), int(ey - y0 + ry * 0.05)), (int(rx * 1.05), int(ry * 0.42)), 0, 10, 170, 1.0,
                    max(2, int(rx * 0.09)), cv2.LINE_AA)
        lash = cv2.GaussianBlur(lash, (0, 0), 0.7)[..., None]
        fill = fill * (1 - lash) + np.array([0.13, 0.08, 0.07], np.float32) * lash
        out[y0:y0 + 2 * R, x0:x0 + 2 * R, :3] = fill * out[y0:y0 + 2 * R, x0:x0 + 2 * R, 3:4]
    return out


def paint_blink(px, closed, ex, ey, rx, ry, b):
    """Lids meet a little below the eye centre: the upper lid comes down, the lower one up a bit."""
    if b < 0.04:
        return
    R = int(max(rx, ry) * 2.2)
    x0, y0 = int(ex - R), int(ey - R)
    S = 3
    m = np.zeros((2 * R * S, 2 * R * S), np.float32)
    cv2.ellipse(m, (int((ex - x0) * S), int((ey - y0) * S)), (int(rx * 1.25 * S), int(ry * 1.45 * S)), 0, 0, 360, 1.0, -1,
                cv2.LINE_AA)
    yy = (np.arange(2 * R * S, dtype=np.float32)[:, None] / S) + y0
    top, bot, line = ey - 1.45 * ry, ey + 1.45 * ry, ey + 0.1 * ry
    up = top + (line - top) * b
    lo = bot - (bot - line) * b
    xx = (np.arange(2 * R * S, dtype=np.float32)[None, :] / S) + x0
    curve = ((xx - ex) / (rx * 1.25)) ** 2 * ry * 0.7          # lid edges follow the eye's curve
    lid = np.clip((up - curve - yy) * 1.2 + 0.5, 0, 1) + np.clip((yy - lo - curve * 0.5) * 1.2 + 0.5, 0, 1)
    m = cv2.resize(np.clip(m * lid, 0, 1), (2 * R, 2 * R), interpolation=cv2.INTER_AREA)[..., None]
    sl = (slice(y0, y0 + 2 * R), slice(x0, x0 + 2 * R))
    px[sl] = px[sl] * (1 - m) + closed[sl] * m


def blink_amount(t, seed):
    """Blinks every ~2.6-4 s (sometimes a double blink), each 0.16 s long."""
    rnd = random.Random(seed)
    tt = rnd.uniform(0.4, 1.6)
    while tt < t + 1:
        for k in ([0, 0.28] if rnd.random() < 0.22 else [0]):
            d = t - (tt + k)
            if 0 <= d < 0.17:
                return math.sin(math.pi * d / 0.17) ** 0.7
        tt += rnd.uniform(2.6, 4.0)
    return 0.0

# ------------------------------------------------------------------ the painted room


def build_room():
    """Warm Indian living room, painted procedurally (world image, float RGB 0..1)."""
    rnd = np.random.default_rng(7)
    yy, xx = np.mgrid[0:WORLD_H, 0:WORLD_W].astype(np.float32)
    img = np.zeros((WORLD_H, WORLD_W, 3), np.float32)
    # wall: warm peach with a soft vertical gradient and plaster texture
    wall = np.array([0.95, 0.80, 0.64], np.float32)
    img[:] = wall * (1.02 - 0.12 * (yy / FLOOR_Y))[..., None]
    tex = cv2.GaussianBlur(rnd.normal(0, 1, (WORLD_H, WORLD_W)).astype(np.float32), (0, 0), 3) * 0.012
    img += tex[..., None]
    # lower wall band (dado) in teal
    band = (yy > FLOOR_Y - 230) & (yy < FLOOR_Y)
    img[band] = np.array([0.36, 0.62, 0.62], np.float32) * (0.95 + 0.05 * np.sin(xx[band] * 0.02))[:, None]
    img[(yy > FLOOR_Y - 236) & (yy <= FLOOR_Y - 230)] = (0.98, 0.93, 0.82)
    # floor: warm wooden planks in perspective towards a vanishing point
    fl = yy >= FLOOR_Y
    depth = np.clip((yy - FLOOR_Y) / (WORLD_H - FLOOR_Y), 0, 1)
    wood = np.array([0.72, 0.47, 0.30], np.float32)
    img[fl] = (wood * (0.86 + 0.2 * depth[fl, None]))
    vx, vy = WORLD_W / 2, 560.0
    ang = np.arctan2(xx - vx, yy - vy)
    planks = (np.sin(ang * 46) > 0.965) & fl
    img[planks] *= 0.82
    rows = fl & (np.sin(np.log(np.clip(yy - vy, 1, None)) * 36) > 0.985)
    img[rows] *= 0.88
    grain = cv2.GaussianBlur(rnd.normal(0, 1, (WORLD_H, WORLD_W)).astype(np.float32), (0, 0), 1.2)
    img[fl] *= (1 + 0.025 * grain[fl])[..., None]
    img[(yy >= FLOOR_Y) & (yy < FLOOR_Y + 14)] = (0.55, 0.36, 0.24)          # skirting board
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")
    # window with sky, frame and curtains
    wx0, wy0, wx1, wy1 = 70, 300, 360, 720
    for i in range(wy1 - wy0):
        u = i / (wy1 - wy0)
        d.line([(wx0, wy0 + i), (wx1, wy0 + i)], fill=(int(150 + 90 * u), int(205 + 30 * u), int(240 - 10 * u)))
    d.ellipse([wx0 + 150, wy0 + 260, wx0 + 420, wy0 + 520], fill=(120, 175, 110))            # trees outside
    d.ellipse([wx0 - 60, wy0 + 300, wx0 + 170, wy0 + 520], fill=(100, 160, 100))
    d.rectangle([wx0, wy0, wx1, wy1], outline=(250, 245, 235), width=16)
    d.line([((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1)], fill=(250, 245, 235), width=10)
    d.line([(wx0, (wy0 + wy1) // 2), (wx1, (wy0 + wy1) // 2)], fill=(250, 245, 235), width=10)
    d.rectangle([wx0 - 20, wy1, wx1 + 20, wy1 + 18], fill=(240, 232, 215))                   # sill
    for side in (0, 1):                                                                      # curtains
        x0 = wx0 - 55 if side == 0 else wx1 - 10
        for k in range(6):
            sh = 0.85 + 0.15 * math.sin(k * 1.9)
            d.rectangle([x0 + k * 11, wy0 - 40, x0 + k * 11 + 11, wy1 + 120],
                        fill=(int(232 * sh), int(140 * sh), int(48 * sh)))
    d.rectangle([wx0 - 80, wy0 - 52, wx1 + 80, wy0 - 38], fill=(120, 80, 50))                # rod
    # marigold toran across the top of the window
    for k in range(26):
        u = k / 25
        x = wx0 - 40 + u * (wx1 - wx0 + 80)
        y = wy0 - 30 + 34 * math.sin(math.pi * ((u * 4) % 1))
        col = (250, 165, 20) if k % 2 else (250, 205, 40)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col)
        if k % 3 == 0:
            d.polygon([(x, y + 6), (x - 7, y + 24), (x + 7, y + 24)], fill=(60, 140, 60))
    # family photo frame
    d.rectangle([600, 540, 790, 690], fill=(120, 75, 45))
    d.rectangle([612, 552, 778, 678], fill=(250, 240, 220))
    for cx_, r_, col in ((650, 22, (90, 110, 80)), (700, 18, (230, 190, 90)), (745, 24, (200, 120, 60))):
        d.ellipse([cx_ - r_ * 0.6, 585, cx_ + r_ * 0.6, 585 + r_ * 1.2], fill=(150, 110, 85))
        d.rectangle([cx_ - r_, 585 + r_ * 1.2, cx_ + r_, 676], fill=col)
    # small shelf with a brass lamp and books
    d.rectangle([560, 800, 820, 814], fill=(125, 82, 52))
    d.rectangle([585, 760, 600, 800], fill=(200, 60, 60)); d.rectangle([602, 768, 616, 800], fill=(60, 110, 190))
    d.rectangle([618, 756, 632, 800], fill=(240, 190, 60))
    d.ellipse([720, 770, 780, 800], fill=(214, 170, 60)); d.polygon([(742, 770), (758, 770), (750, 742)], fill=(255, 190, 60))
    # potted plant in the right corner
    d.rectangle([770, 900, 860, 1010], fill=(170, 90, 50))
    for k in range(9):
        a_ = -2.3 + k * 0.45
        x2, y2 = 815 + 120 * math.cos(a_), 900 + 130 * math.sin(a_)
        d.polygon([(815, 905), (x2 - 12, y2), (x2 + 12, y2 + 6)], fill=(60 + 10 * k, 140 + 6 * k, 70))
    # round rug under the family
    rug = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(rug)
    cx_, cy_ = 450, 1300
    for i, (col, s_) in enumerate([((150, 40, 50, 255), 1.0), ((230, 170, 60, 255), 0.9), ((150, 40, 50, 255), 0.82),
                                   ((40, 110, 120, 255), 0.66), ((230, 170, 60, 255), 0.45), ((150, 40, 50, 255), 0.3)]):
        dr.ellipse([cx_ - 380 * s_, cy_ - 105 * s_, cx_ + 380 * s_, cy_ + 105 * s_], fill=col)
    pil.alpha_composite(rug) if pil.mode == "RGBA" else pil.paste(rug, (0, 0), rug)
    img = np.asarray(pil).astype(np.float32) / 255.0
    # wall clock (face drawn here, hands every frame)
    cv2.circle(img, (690, 380), 62, (0.45, 0.30, 0.20), -1, cv2.LINE_AA)
    cv2.circle(img, (690, 380), 54, (0.99, 0.97, 0.92), -1, cv2.LINE_AA)
    for k in range(12):
        a_ = k * math.pi / 6
        cv2.circle(img, (int(690 + 44 * math.cos(a_)), int(380 + 44 * math.sin(a_))), 3, (0.3, 0.25, 0.2), -1, cv2.LINE_AA)
    # sunlight: soft beam from the window and a warm patch on the floor
    light = np.zeros((WORLD_H, WORLD_W), np.float32)
    cv2.fillPoly(light, [np.array([[wx0, wy0], [wx1, wy0], [wx1 + 330, 1250], [wx0 + 150, 1330]], np.int32)], 1.0)
    light = cv2.GaussianBlur(light, (0, 0), 40)
    img += light[..., None] * np.array([0.13, 0.10, 0.05], np.float32)
    patch = np.zeros_like(light)
    cv2.fillPoly(patch, [np.array([[250, 1060], [560, 1060], [700, 1250], [330, 1250]], np.int32)], 1.0)
    img += cv2.GaussianBlur(patch, (0, 0), 25)[..., None] * np.array([0.10, 0.08, 0.03], np.float32)
    # gentle ambient occlusion where the wall meets the floor
    ao = np.exp(-np.abs(yy - FLOOR_Y) / 26)[..., None] * 0.12
    img *= (1 - ao)
    return np.clip(img, 0, 1)


def clock_hands(canvas, cam, t):
    z, cx, cy = cam
    sx, sy = W / 2 + z * (690 - cx), H / 2 + z * (380 - cy)
    for length, speed, th in ((30, 0.05, 4), (44, 0.6, 3)):
        a = -math.pi / 2 + 2.1 + t * speed
        cv2.line(canvas, (int(sx), int(sy)), (int(sx + z * length * math.cos(a)), int(sy + z * length * math.sin(a))),
                 (0.2, 0.15, 0.12), max(1, int(th * z)), cv2.LINE_AA)


def room_view(rooms, cam, blur_level=0):
    z, cx, cy = cam
    m = np.array([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]], np.float32)
    return cv2.warpAffine(rooms[blur_level], m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ------------------------------------------------------------------ props: jars, coins, note, table


def coin_draw(canvas, x, y, rx, ry, s=1.0):
    """Gold coin; ry < rx means it is lying flat or turning."""
    th = max(1, int(5 * s * (1 - ry / max(rx, 1)) + 1))
    cv2.ellipse(canvas, (int(x), int(y + th)), (int(rx), int(max(1, ry))), 0, 0, 360, (0.62, 0.40, 0.08), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(x), int(y)), (int(rx), int(max(1, ry))), 0, 0, 360, (0.98, 0.76, 0.22), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(x), int(y)), (int(rx * 0.74), int(max(1, ry * 0.74))), 0, 0, 360, (0.85, 0.58, 0.12),
                max(1, int(2 * s)), cv2.LINE_AA)
    cv2.ellipse(canvas, (int(x - rx * 0.3), int(y - ry * 0.35)), (max(1, int(rx * 0.25)), max(1, int(ry * 0.2))), 0, 0, 360,
                (1.0, 0.95, 0.75), -1, cv2.LINE_AA)


class CoinSim:
    """2D physics for flat coins dropping into jars: gravity, bounces, friction, stacking.

    Coins are horizontal capsules (they land flat). Positions are in screen
    pixels of the jars shot."""
    G = 3200.0
    R = 9.0          # half thickness
    HALF = 18.0      # half length of the flat part

    def __init__(self, jars, drops, t_end):
        self.jars = jars                          # list of dicts: cx, top, bottom, inner_half
        self.drops = sorted(drops)                # (time, jar index, seed)
        self.frames, self.events = [], []         # per frame: list of coin states; events: (t, kind, jar)
        dt = 1 / (FPS * 10)
        coins = []
        nd = 0
        t = 0.0
        n_steps = int(t_end * FPS * 10) + 1
        for step in range(n_steps):
            t = step * dt
            while nd < len(self.drops) and self.drops[nd][0] <= t:
                td, j, seed = self.drops[nd]
                r = random.Random(seed)
                jar = jars[j]
                coins.append({"x": jar["cx"] + r.uniform(-16, 16), "y": -70.0, "vx": r.uniform(-30, 30), "vy": 250.0,
                              "jar": j, "spin": r.uniform(0, 6.28), "sr": r.uniform(14, 20), "landed": None, "t0": td})
                nd += 1
            for c in coins:
                c["vy"] += self.G * dt
                c["x"] += c["vx"] * dt
                c["y"] += c["vy"] * dt
                jar = jars[c["jar"]]
                if c["y"] > jar["top"]:                                       # inside the glass
                    lo = jar["cx"] - jar["inner_half"] + self.HALF + self.R
                    hi = jar["cx"] + jar["inner_half"] - self.HALF - self.R
                    if c["x"] < lo:
                        c["x"], c["vx"] = lo, abs(c["vx"]) * 0.4
                    if c["x"] > hi:
                        c["x"], c["vx"] = hi, -abs(c["vx"]) * 0.4
                    floor = jar["bottom"] - self.R
                    if c["y"] > floor:
                        self._hit(c, t, abs(c["vy"]))
                        c["y"] = floor
                        c["vy"] = -c["vy"] * 0.28 if c["vy"] > 180 else 0.0
                        c["vx"] *= 0.7
            for _ in range(2):                                                # coin-coin contacts
                for i in range(len(coins)):
                    a = coins[i]
                    for k in range(i + 1, len(coins)):
                        b = coins[k]
                        if a["jar"] != b["jar"]:
                            continue
                        ddx = b["x"] - a["x"]
                        gap = max(0.0, abs(ddx) - 2 * self.HALF)
                        ddy = b["y"] - a["y"]
                        dist = math.hypot(gap, ddy)
                        if dist >= 2 * self.R or dist < 1e-6:
                            if dist < 1e-6 and abs(ddx) < 2 * self.HALF + 2 * self.R:
                                ddy, dist, gap = -1.0, 1.0, 0.0
                            else:
                                continue
                        nx = (math.copysign(gap, ddx)) / dist
                        ny = ddy / dist
                        pen = 2 * self.R - dist
                        wa = 0.0 if a["landed"] is not None and a["vy"] == 0 and ny > 0 else 0.5
                        wb = 1.0 - wa
                        a["x"] -= nx * pen * wa
                        a["y"] -= ny * pen * wa
                        b["x"] += nx * pen * wb
                        b["y"] += ny * pen * wb
                        rel = (b["vx"] - a["vx"]) * nx + (b["vy"] - a["vy"]) * ny
                        if rel < 0:
                            imp = -1.3 * rel
                            a["vx"] -= imp * nx * wa
                            a["vy"] -= imp * ny * wa
                            b["vx"] += imp * nx * wb
                            b["vy"] += imp * ny * wb
                            upper = b if ny > 0 else a
                            for c_ in (a, b):
                                self._hit(c_, t, abs(rel) if c_ is upper else 0.0)
                            upper["vx"] *= 0.85
            if step % 10 == 0:
                self.frames.append([(c["x"], c["y"], c["jar"], c["spin"] + c["sr"] * (t - c["t0"]),
                                     None if c["landed"] is None else t - c["landed"]) for c in coins])

    def _hit(self, c, t, speed):
        if c["landed"] is None:
            c["landed"] = t
            self.events.append((t, "coin", c["jar"]))
        elif speed > 350:
            self.events.append((t, "clink", c["jar"]))

    def state(self, t):
        i = int(clamp(t * FPS, 0, len(self.frames) - 1))
        return self.frames[i]

    def count(self, j, t):
        return sum(1 for (te, k, jj) in self.events if k == "coin" and jj == j and te <= t)

    def last_landing(self, j, t):
        ts = [te for (te, k, jj) in self.events if k == "coin" and jj == j and te <= t]
        return ts[-1] if ts else None


def draw_jar_back(canvas, cx, top, bottom, half, s=1.0):
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(top), int(bottom)
    if x1 <= 0 or x0 >= W:
        return
    reg = canvas[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    reg *= np.array([0.80, 0.86, 0.90], np.float32)
    ellipse_shadow(canvas, cx, bottom + 4 * s, half * 1.05, 12 * s, 0.5, 6 * s)


def draw_jar_front(canvas, cx, top, bottom, half, color, label_img, s=1.0):
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(top), int(bottom)
    lay = np.zeros((H, W), np.float32)
    # glass highlights: a broad stripe on the left, a thin line on the right, the rim and the thick bottom
    cv2.rectangle(lay, (int(x0 + 10 * s), y0 + int(30 * s)), (int(x0 + 24 * s), y1 - int(20 * s)), 0.45, -1, cv2.LINE_AA)
    cv2.rectangle(lay, (int(x1 - 14 * s), y0 + int(40 * s)), (int(x1 - 9 * s), y1 - int(30 * s)), 0.3, -1, cv2.LINE_AA)
    cv2.rectangle(lay, (x0, y0), (x1, y1), 0.5, max(1, int(3 * s)), cv2.LINE_AA)
    cv2.ellipse(lay, (int(cx), int(y1 - 6 * s)), (int(half - 4 * s), int(9 * s)), 0, 0, 360, 0.35, -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.6 * s)
    canvas += lay[..., None] * np.array([0.9, 0.95, 1.0], np.float32) * (1 - canvas) * 0.9
    # coloured neck band with the label
    col = np.array(color, np.float32) / 255.0
    by0, by1 = int(y0 - 10 * s), int(y0 + 34 * s)
    cv2.rectangle(canvas, (int(x0 - 6 * s), by0), (int(x1 + 6 * s), by1), tuple(float(v) * 0.75 for v in col), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (int(x0 - 6 * s), by0), (int(x1 + 6 * s), by1 - int(6 * s)), tuple(float(v) for v in col), -1, cv2.LINE_AA)
    if label_img is not None:
        lh, lw = label_img.shape[:2]
        sc = min(1.0, (2 * half - 8 * s) / lw) * 1.0
        blit(canvas, label_img, affine((lw / 2, lh / 2), (cx, (by0 + by1 - 6 * s) / 2), sc, sc))


def draw_table(canvas, x0, x1, top_y, front_y, legs_to):
    cv2.rectangle(canvas, (int(x0 + 20), int(front_y)), (int(x0 + 40), int(legs_to)), (0.40, 0.24, 0.14), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (int(x1 - 40), int(front_y)), (int(x1 - 20), int(legs_to)), (0.40, 0.24, 0.14), -1, cv2.LINE_AA)
    pts = np.array([[x0 + 18, top_y], [x1 - 18, top_y], [x1, front_y], [x0, front_y]], np.int32)
    cv2.fillPoly(canvas, [pts], (0.66, 0.42, 0.25), cv2.LINE_AA)
    cv2.rectangle(canvas, (int(x0), int(front_y)), (int(x1), int(front_y + 16)), (0.50, 0.30, 0.17), -1, cv2.LINE_AA)


def rupee_note(w=150):
    h = int(w * 0.48)
    im = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([4, 4, w + 4, h + 4], radius=8, fill=(168, 150, 205), outline=(110, 90, 150), width=3)
    d.ellipse([w * 0.62, h * 0.18, w * 0.62 + h * 0.62, h * 0.8], outline=(120, 100, 160), width=3)
    f = ImageFont.truetype(FONT, int(h * 0.42))
    d.text((12, h * 0.22), "₹100", font=f, fill=(70, 50, 110))
    return premul(np.asarray(im))

# ------------------------------------------------------------------ timeline


class Timeline:
    PRE = {"b1": 0.55}
    POST = {"b9": 2.6}

    def __init__(self, show, voice):
        self.beats = []
        t = 0.0
        for b in show["beats"]:
            v = voice.get(b["id"]) or estimate(b["say"])
            pre = self.PRE.get(b["id"], 0.18)
            post = self.POST.get(b["id"], 0.32)
            if b["shot"] == "jars":
                pre = max(pre, 0.3)
                post = max(post, 0.45 + 0.36 * max(0, b.get("coins", 0) - 1) + 1.1 - pre - v["dur"])
            beat = dict(b, t0=t, speak=t + pre, dur_voice=v["dur"], t1=t + pre + v["dur"] + post,
                        words=v["words"], env=v["env"])
            self.beats.append(beat)
            t = beat["t1"]
        self.total = t
        self.shots = []
        for b in self.beats:
            if self.shots and self.shots[-1]["shot"] == b["shot"] and b["shot"] == "jars":
                self.shots[-1]["beats"].append(b)
                self.shots[-1]["t1"] = b["t1"]
            else:
                self.shots.append({"shot": b["shot"], "t0": b["t0"], "t1": b["t1"], "beats": [b]})

    def beat_at(self, t):
        for b in self.beats:
            if b["t0"] <= t < b["t1"]:
                return b
        return self.beats[-1]

    def shot_at(self, t):
        for s in self.shots:
            if s["t0"] <= t < s["t1"]:
                return s
        return self.shots[-1]

    def loud(self, t, who=None):
        """Voice loudness (0..1) at global time t, for one speaker or anyone."""
        b = self.beat_at(t)
        if who and b["who"] != who:
            return 0.0
        u = (t - b["speak"]) * 30
        if u < 0 or u >= len(b["env"]) - 1:
            return 0.0
        i = int(u)
        f = u - i
        return b["env"][i] * (1 - f) + b["env"][i + 1] * f


def estimate(text):
    """Timing guess when no voice file exists (silent draft): ~2.6 words/s."""
    words, t = [], 0.0
    for w in text.split():
        d = 0.16 + 0.055 * len(w.strip(",.!?"))
        words.append([round(t, 3), round(t + d, 3), w])
        t += d + (0.25 if w[-1] in ".!?," else 0.05)
    dur = t
    env, r = [], random.Random(len(text))
    for i in range(int(dur * 30) + 1):
        tt = i / 30
        on = any(a <= tt <= b for a, b, _ in words)
        env.append(round((0.45 + 0.55 * abs(math.sin(tt * 13.0 + r.random()))) if on else 0.0, 3))
    return {"dur": dur, "words": words, "env": env}

# ------------------------------------------------------------------ captions and title pill


class Captions:
    def __init__(self, tl):
        self.chunks = []
        for b in tl.beats:
            ws = b["words"]
            cur = []
            for i, (a, e, w) in enumerate(ws):
                cur.append((b["speak"] + a, b["speak"] + e, w))
                if len(cur) == 3 or w[-1] in ".!?," or i == len(ws) - 1:
                    self.chunks.append(cur)
                    cur = []
        for i, c in enumerate(self.chunks):
            nxt = self.chunks[i + 1][0][0] if i + 1 < len(self.chunks) else c[-1][1] + 0.6
            c_end = min(nxt, c[-1][1] + 0.6)
            self.chunks[i] = (c[0][0] - 0.05, c_end, c)
        self.cache = {}

    def image(self, t):
        for k, (a, b, ws) in enumerate(self.chunks):
            if a <= t < b:
                hi = 0
                for j, (wa, we, _) in enumerate(ws):
                    if t >= wa:
                        hi = j
                key = (k, hi)
                if key not in self.cache:
                    self.cache[key] = self._render([w.upper() for _, _, w in ws], hi)
                return self.cache[key], t - a
        return None, 0

    def _render(self, words, hi):
        f = ImageFont.truetype(FONT, 64)
        space = 18
        sizes = [f.getbbox(w, stroke_width=8) for w in words]
        tw = sum(s[2] - s[0] for s in sizes) + space * (len(words) - 1)
        th = max(s[3] for s in sizes) + 10
        im = Image.new("RGBA", (tw + 20, th + 20), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        x = 10
        for w, s, i in zip(words, sizes, range(len(words))):
            d.text((x - s[0], 10), w, font=f, fill=(255, 214, 40) if i == hi else (255, 255, 255),
                   stroke_width=8, stroke_fill=(15, 15, 20))
            x += s[2] - s[0] + space
        return premul(np.asarray(im))


def title_pill(ep, title):
    f = ImageFont.truetype(FONT, 26)
    chip = f"EP {ep}"
    cw = f.getbbox(chip)[2] + 24
    tw = f.getbbox(title)[2]
    w, h = 64 + cw + tw + 30, 54
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=27, fill=(20, 20, 30, 150))
    d.ellipse([8, 8, 46, 46], fill=(250, 196, 40), outline=(190, 130, 20), width=3)
    d.text((18, 9), "₹", font=ImageFont.truetype(FONT, 26), fill=(130, 80, 10))
    d.rounded_rectangle([56, 10, 56 + cw, 44], radius=12, fill=(255, 214, 40))
    d.text((68, 11), chip, font=f, fill=(30, 30, 30))
    d.text((56 + cw + 14, 11), title, font=f, fill=(255, 255, 255))
    return premul(np.asarray(im))

# ------------------------------------------------------------------ the show


class Show:
    def __init__(self, show_dir, voice):
        self.dir = Path(show_dir)
        self.cfg = yaml.safe_load((self.dir / "show.yaml").read_text(encoding="utf-8"))
        self.tl = Timeline(self.cfg, voice)
        self.cast = Cast()
        room = build_room()
        self.rooms = [room, cv2.GaussianBlur(room, (0, 0), 2.2), cv2.GaussianBlur(room, (0, 0), 7)]
        self.caps = Captions(self.tl)
        self.pill = title_pill(self.cfg.get("ep", 1), self.cfg.get("title", ""))
        self.note = rupee_note()
        jc = self.cfg["jars"]
        self.jar_labels = [text_image(j["label"], 30, fill=(255, 255, 255)) for j in jc]
        self.jar_colors = [j["color"] for j in jc]
        self.sfx = []
        self._setup_jars()
        self._setup_busts()
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.vignette = (1 - 0.32 * (((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.72)) ** 2))[..., None]
        self.rays = self._rays()
        for s in self.tl.shots[1:]:
            self.sfx.append((s["t0"], "whoosh"))

    # ---- setup -------------------------------------------------------------
    def _setup_jars(self):
        shot = next(s for s in self.tl.shots if s["shot"] == "jars")
        self.jar_shot = shot
        self.jars = [{"cx": cx, "top": 690.0, "bottom": 1010.0, "inner_half": 72.0} for cx in (140.0, 360.0, 580.0)]
        drops = []
        for b in shot["beats"]:
            for k in range(b.get("coins", 0)):
                drops.append((b["t0"] - shot["t0"] + 0.45 + 0.36 * k, b["jar"], (sum(map(ord, b["id"])) * 131 + k * 17) % 10000))
        self.sim = CoinSim(self.jars, drops, shot["t1"] - shot["t0"] + 0.1)
        for (te, kind, j) in self.sim.events:
            self.sfx.append((shot["t0"] + te, kind))
        self.final_coins = self.sim.state(1e9)

    def _setup_busts(self):
        self.busts = {}
        for b in self.tl.beats:
            if b["shot"] == "close":
                key = (b["who"], b["face"])
                if key not in self.busts:
                    self.busts[key] = self._make_bust(*key)

    def _make_bust(self, char, face):
        """Head-and-shoulders puppet for close-ups: the expression sprite plus a body below it."""
        target_w = 610 if char == "dad" else 520
        scale = target_w / self.cast.size(char, face)[0]
        px = self.cast.get(char, face, scale)
        h, w = px.shape[:2]
        ext = 1100
        BW = int(w * 1.7)
        out = np.zeros((h + ext, BW, 4), np.float32)
        ox = (BW - w) // 2
        if char == "dad":
            # continue the shirt below the crop: the bottom rows, smoothed, darker lower down
            row = px[h - 16:h - 10].mean(axis=0)
            rowc = row[:, :3] / np.clip(row[:, 3:4], 1e-3, None)
            rowc = cv2.GaussianBlur(rowc[None], (0, 0), 9)[0]
            body = np.zeros((ext + 8, w, 4), np.float32)
            g = np.linspace(1.0, 0.82, ext + 8, dtype=np.float32)[:, None, None]
            cols = (row[:, 3] > 0.35).astype(np.float32)
            cols = cv2.GaussianBlur(cols[None], (0, 0), 1.5)[0]
            body[..., :3] = rowc[None] * g * cols[None, :, None]
            body[..., 3] = cols[None, :]
            xs_ = np.linspace(-1, 1, w, dtype=np.float32)[None, :, None]
            body[..., :3] *= (1.0 - 0.18 * xs_ ** 4)
            body[:, w // 2 - 3:w // 2 + 3, :3] *= 0.8                     # button placket
            for k in range(1, 4):
                cv2.circle(body, (w // 2, 70 * k), 5, (0.30, 0.33, 0.24, 1.0), -1, cv2.LINE_AA)
            out[h - 18:h - 10 + ext, ox:ox + w] = body
            out[0:h, ox:ox + w] = px[..., :] + out[0:h, ox:ox + w] * (1 - px[..., 3:4])
        else:
            out[0:h, ox:ox + w] = px
            # white T-shirt with a round collar, drawn over the bottom of the neck
            tee = np.zeros((h + ext, BW), np.float32)
            ny = int(h * 0.90)
            cxp = BW // 2
            pts = np.array([[cxp - int(w * 0.2), ny], [cxp - int(w * 0.5), ny + int(h * 0.1)],
                            [cxp - int(w * 0.64), ny + int(h * 0.36)], [cxp - int(w * 0.66), h + ext],
                            [cxp + int(w * 0.66), h + ext], [cxp + int(w * 0.64), ny + int(h * 0.36)],
                            [cxp + int(w * 0.5), ny + int(h * 0.1)], [cxp + int(w * 0.2), ny]], np.int32)
            cv2.fillPoly(tee, [pts], 1.0, cv2.LINE_AA)
            neck = np.zeros_like(tee)
            cv2.ellipse(neck, (cxp, ny - 2), (int(w * 0.2), int(h * 0.07)), 0, 0, 180, 1.0, -1, cv2.LINE_AA)
            tee = np.clip(tee - neck, 0, 1)
            tee = cv2.GaussianBlur(tee, (0, 0), 1.0)
            xx = np.linspace(-1, 1, BW, dtype=np.float32)[None, :]
            shade = (0.97 - 0.3 * np.clip(np.abs(xx) / 0.66, 0, 1) ** 3 - 0.06 * np.clip(xx, 0, 1))
            tcol = np.dstack([shade * 0.97, shade * 0.97, shade * 0.99])
            collar = np.zeros_like(tee)
            cv2.ellipse(collar, (cxp, ny - 2), (int(w * 0.215), int(h * 0.08)), 0, 0, 180, 1.0, max(2, int(w * 0.03)), cv2.LINE_AA)
            tcol = tcol * (1 - 0.12 * collar[..., None])
            a = tee[..., None]
            out[..., :3] = tcol * a + out[..., :3] * (1 - a)
            out[..., 3:4] = a + out[..., 3:4] * (1 - a)
        f = FACES[(char, face)]
        sc = scale
        eyes = [(ox + x * sc, y * sc) for x, y in f["eyes"]]
        closed = make_closed_eyes(out, eyes, f["erx"] * sc, f["ery"] * sc)
        return {"px": out, "closed": closed, "scale": sc, "ox": ox,
                "eyes": [(ox + x * sc, y * sc) for x, y in f["eyes"]], "erx": f["erx"] * sc, "ery": f["ery"] * sc,
                "mouth": (ox + f["mouth"][0] * sc, f["mouth"][1] * sc), "mw": f["mw"] * sc, "h": h}

    def _rays(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        return np.arctan2(yy - H * 0.55, xx - W / 2), np.hypot(yy - H * 0.55, xx - W / 2)

    # ---- characters in the room --------------------------------------------
    def draw_char(self, canvas, cam, char, name, x, y, face_dir, lift=0.0, sx=1.0, sy=1.0, rot=0.0,
                  hrot=0.0, hdx=0.0, hdy=0.0, lean=0.0, alpha=1.0, shadow=True):
        z, cx, cy = cam
        px = self.cast.get(char, name)
        flip = FACING[char].get(name, 1) != face_dir
        px2, anchor = rig(px, CHARS[char]["neck"], hrot, hdx, hdy, lean)
        if flip:
            px2 = np.ascontiguousarray(px2[:, ::-1])
        X, Y = W / 2 + z * (x - cx), H / 2 + z * (y - cy)
        if shadow:
            hgt = px2.shape[0]
            wdt = px2.shape[1]
            k = 1.0 - 0.5 * clamp(lift / 220)
            ellipse_shadow(canvas, X, Y + 4 * z, wdt * 0.32 * z * k, 13 * z * k, 0.55 * k, 7 * z)
            # long soft cast shadow away from the window (to the right, onto the floor)
            m = np.array([[z * sx, 0.95 * z, 0], [0, 0.2 * z, 0]], np.float32)
            m[:, 2] = np.array([X, Y + lift * z * 0.1], np.float32) - m[:, :2] @ np.array([anchor[0], hgt], np.float32)
            blit_shadow(canvas, px2[..., 3].copy(), m, 0.28 * (1 - 0.6 * clamp(lift / 200)), 6 * z)
        m = affine(anchor, (X, Y - lift * z), sx * z, sy * z, rot)
        blit(canvas, px2, m, alpha)

    # ---- frame -------------------------------------------------------------
    def frame(self, t):
        shot = self.tl.shot_at(t)
        u = t - shot["t0"]
        d = shot["t1"] - shot["t0"]
        kind = shot["shot"]
        punch = 1.0 + 0.06 * (1 - ease_out(u / 0.3))                 # punch-in on every cut
        if kind == "jars":
            canvas = self.shot_jars(t, u, d, punch)
        elif kind == "close":
            canvas = self.shot_close(t, u, d, shot["beats"][0], punch)
        else:
            canvas = self.shot_wide(t, u, d, kind, punch)
        canvas *= self.vignette
        # captions: close-ups above the face, otherwise the lower third (nudged left of the Shorts buttons)
        img, age = self.caps.image(t)
        if img is not None:
            ch, cw = img.shape[:2]
            sc = min(1.0, 600 / cw) * (0.86 + 0.14 * back_out(age / 0.18))
            cy = 250 if kind in ("close", "jars") else 1020
            blit(canvas, img, affine((cw / 2, ch / 2), (340, cy), sc, sc))
        ph, pw = self.pill.shape[:2]
        blit(canvas, self.pill, affine((pw / 2, 0), (W / 2, 60)))
        rng = np.random.default_rng(int(t * FPS))
        canvas += rng.normal(0, 0.012, (H // 2, W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
        return np.clip(canvas, 0, 1)

    def shot_wide(self, t, u, d, kind, punch):
        beat = self.tl.beat_at(t)
        push = 1.14 + 0.05 * ease_in_out(u / d)
        z = push * punch
        cam = (z, CAM0[0], CAM0[1] + 95 + 60 * (push - 1.14))
        canvas = room_view(self.rooms, cam, 1)
        clock_hands(canvas, cam, t)
        # small table with the jars between them (empty before the lesson, full after)
        self.draw_mini_jars(canvas, cam, full=(kind != "wide_run_in"))
        dad_talk = self.tl.loud(t, "dad")
        kid_talk = self.tl.loud(t, "aarav")
        sp = beat["speak"] - beat["t0"]
        # ---------------- Aarav
        if kind == "wide_run_in":
            arrive = 1.15
            if u < arrive:
                k = ease_out(u / arrive)
                x = -140 + (250 + 140) * k
                step = abs(math.sin(2 * math.pi * 2.8 * u))
                self.draw_char(canvas, cam, "aarav", "pose_run", x, FEET_Y, 1, lift=step * 16,
                               sx=1 - 0.03 * step, sy=1 + 0.04 * step, rot=4 * math.sin(2 * math.pi * 2.8 * u))
                self.sfx_once(t, u, [0.18, 0.36, 0.54, 0.72, 0.9], "step")
            else:
                self.kid_jump(canvas, cam, u - arrive, 250, t)
                self.sfx_once(t, u, [arrive + 0.38], "boing")
        elif kind == "wide_end":
            # two happy jumps, then a bouncy idle
            v = u - sp
            if 0.2 < v < 1.6:
                self.kid_jump(canvas, cam, (v - 0.2) % 0.7 / 0.7 * 1.25, 250, t, small=True)
            else:
                self.kid_idle(canvas, cam, t, 250, kid_talk, look=1)
        else:
            self.kid_idle(canvas, cam, t, 250, kid_talk, look=1, nod=True)
        # ---------------- Dad
        if kind == "wide_teach":
            v = u - sp
            pose = "turn_3q"
            if 0.15 < v < 2.6:
                pose = "pose_point"
            elif 2.6 <= v < 4.6:
                pose = "pose_wave"
            self.dad_pose(canvas, cam, t, pose, dad_talk, v, [0.15, 2.6, 4.6])
        elif kind == "wide_end":
            v = u - sp
            pose = "pose_wave" if v > 0.1 else "turn_3q"
            self.dad_pose(canvas, cam, t, pose, dad_talk, v, [0.1])
        else:
            react = clamp((u - 1.6) / 0.3)
            self.dad_pose(canvas, cam, t, "turn_3q", dad_talk, u, [], lean=-14 * react)
        # the hundred rupee note pops above Aarav's head
        if kind == "wide_run_in" and u > 1.55:
            pop = back_out((u - 1.55) / 0.35)
            nh, nw = self.note.shape[:2]
            X = W / 2 + z * (250 - cam[1])
            Y = H / 2 + z * (FEET_Y - 600 - cam[2]) - 8 * math.sin(3 * u)
            blit(canvas, self.note, affine((nw / 2, nh / 2), (X + 90 * z, Y - 40 * z), pop * z, pop * z, 6 * math.sin(2.2 * u)))
            self.sfx_once(t, u, [1.55], "pop")
        if kind == "wide_end":
            self.end_card(canvas, t, u - sp - beat["dur_voice"] + 0.4)
        return canvas

    def kid_jump(self, canvas, cam, v, x, t, small=False):
        """Anticipation squash, stretch on take-off, parabola in the air, springy landing."""
        hgt = 120 if small else 190
        if v < 0.18:                                                      # squash down
            k = smooth(v / 0.18)
            self.draw_char(canvas, cam, "aarav", "turn_3q", x, FEET_Y, 1, sx=1 + 0.08 * k, sy=1 - 0.1 * k)
        elif v < 0.3:                                                     # stretch, leaving the floor
            k = (v - 0.18) / 0.12
            self.draw_char(canvas, cam, "aarav", "pose_jump", x, FEET_Y, 1, lift=k * 40, sx=0.92, sy=1.12)
        elif v < 0.9:                                                     # air time
            k = (v - 0.3) / 0.6
            lift = 40 + 4 * k * (1 - k) * hgt - 40 * k
            self.draw_char(canvas, cam, "aarav", "pose_jump", x, FEET_Y, 1, lift=lift,
                           sx=0.97, sy=1.04, rot=-4 * (k - 0.5))
        else:                                                             # land and settle
            s = spring(v - 0.9, 0.14, 2.4, 6.5)
            self.draw_char(canvas, cam, "aarav", "turn_3q", x, FEET_Y, 1, sx=1 + s * 0.8, sy=1 - s,
                           hdy=6 * max(0, s) * 10)

    def kid_idle(self, canvas, cam, t, x, talk, look=1, nod=False):
        br = math.sin(2 * math.pi * 0.45 * t)
        hn = 0.0
        if nod:                                                           # listening: small nods
            hn = 5 * max(0, math.sin(2 * math.pi * 0.7 * t)) ** 4
        self.draw_char(canvas, cam, "aarav", "turn_3q", x, FEET_Y, look, sx=1 - 0.004 * br, sy=1 + 0.01 * br,
                       hrot=3 * noise1(t, 2) + 4 * talk, hdy=hn + 4 * talk, hdx=2 * noise1(t * 0.7, 5),
                       lean=4 * noise1(t * 0.5, 1))

    def dad_pose(self, canvas, cam, t, pose, talk, v, switches, lean=0.0):
        """Dad on the right, facing Aarav. Pose changes pop with squash and overshoot."""
        since = min([v - s for s in switches if v >= s] or [9])
        s = spring(since, 0.07, 2.2, 7)
        br = math.sin(2 * math.pi * 0.35 * t + 1)
        nod = 6 * talk * (0.6 + 0.4 * math.sin(2 * math.pi * 2.1 * t))
        self.draw_char(canvas, cam, "dad", pose, 640, FEET_Y + 6, -1, sx=(1 + 0.5 * s) * (1 - 0.003 * br),
                       sy=(1 - s) * (1 + 0.008 * br), hrot=-2.5 * noise1(t, 7) - 3 * talk, hdy=nod,
                       hdx=-3 * noise1(t * 0.6, 3), lean=lean - 6 * talk + 3 * noise1(t * 0.4, 9))

    def sfx_once(self, t, u, times, kind):
        pass                                                              # collected up front (see collect_sfx)

    def draw_mini_jars(self, canvas, cam, full):
        z, cx, cy = cam
        s = 0.5 * z
        tx, ty = W / 2 + z * (450 - cx), H / 2 + z * (1180 - cy)
        draw_table(canvas, tx - 170 * z, tx + 170 * z, ty - 20 * z, ty + 18 * z, ty + 110 * z)
        for j in range(3):
            jcx = tx + (j - 1) * 105 * z
            top, bottom = ty - 115 * z, ty
            draw_jar_back(canvas, jcx, top, bottom, 72 * s, s)
            if full:
                for (x, y, jj, spin, landed) in self.final_coins:
                    if jj == j:
                        X = jcx + (x - self.jars[j]["cx"]) * s
                        Y = bottom + (y - self.jars[j]["bottom"]) * s
                        coin_draw(canvas, X, Y, 27 * s, 9 * s, s)
            draw_jar_front(canvas, jcx, top, bottom, 72 * s, self.jar_colors[j], self.jar_labels[j], s)

    def end_card(self, canvas, t, v):
        if v < 0:
            return
        labels = self.tl.beats[-1].get("card") or []
        for i, lab in enumerate(labels):
            k = back_out((v - 0.25 * i) / 0.35)
            if k <= 0.01:
                continue
            key = ("card", i)
            if key not in self.cast.cache:
                f = ImageFont.truetype(FONT, 44)
                tw = f.getbbox(lab)[2]
                im = Image.new("RGBA", (tw + 60, 84), (0, 0, 0, 0))
                dd = ImageDraw.Draw(im)
                c = tuple(self.jar_colors[i])
                dd.rounded_rectangle([0, 0, tw + 59, 83], radius=40, fill=c + (255,), outline=(255, 255, 255, 255), width=5)
                dd.text((30, 12), lab, font=f, fill=(255, 255, 255), stroke_width=3, stroke_fill=(40, 40, 40))
                self.cast.cache[key] = premul(np.asarray(im))
            img = self.cast.cache[key]
            ih, iw = img.shape[:2]
            blit(canvas, img, affine((iw / 2, ih / 2), (W / 2, 200 + 100 * i), k, k, 3 * (1 - k)))

    def shot_close(self, t, u, d, beat, punch):
        who, face = beat["who"], beat["face"]
        bust = self.busts[(who, face)]
        z = (1.0 + 0.04 * ease_in_out(u / d)) * punch
        if face == "ex_happy" and who == "aarav":
            canvas = self.ray_burst(t)
        else:
            focus = (640, 650) if who == "dad" else (260, 860)
            canvas = room_view(self.rooms, (1.7, focus[0], focus[1]), 2) * 0.96
        px = bust["px"].copy()
        talk = self.tl.loud(t, who)
        lt = t - beat["speak"]
        open_ = clamp((talk - 0.1) * 1.5) if lt > 0 else 0.0
        wide = 0.5 + 0.5 * math.sin(lt * 9.1 + 0.6 * math.sin(lt * 3.3))
        mx, my = bust["mouth"]
        paint_mouth(px, mx, my, bust["mw"], open_, wide)
        b = blink_amount(t, sum(map(ord, who)))
        for ex, ey in bust["eyes"]:
            paint_blink(px, bust["closed"], ex, ey, bust["erx"], bust["ery"], b)
        ph, pw = px.shape[:2]
        rot = 1.6 * noise1(t * 0.8, 3) + 2.0 * talk * math.sin(lt * 4.0)
        bob = 5 * talk + 3 * math.sin(2 * math.pi * 0.4 * t)
        face_y = bust["mouth"][1]
        X = W / 2 + 8 * noise1(t * 0.5, 4)
        Y = H + 40 + bob                                                  # bottom of the bust below the frame
        pop = back_out(u / 0.28)
        sc = z * (0.94 + 0.06 * pop) * (1 + 0.012 * talk)
        # place the face around 46% of the height
        anchor_y = ph
        target_face = H * 0.6
        lift = (Y - target_face) - (ph - face_y) * sc
        blit(canvas, px, affine((pw / 2, anchor_y), (X, Y - lift), sc, sc, rot))
        return canvas

    def ray_burst(self, t):
        ang, rad = self.rays
        k = (np.sin(ang * 12 + t * 1.5) > 0).astype(np.float32)
        c1 = np.array([1.0, 0.80, 0.25], np.float32)
        c2 = np.array([1.0, 0.65, 0.20], np.float32)
        img = c1 * k[..., None] + c2 * (1 - k[..., None])
        img *= (1.05 - 0.35 * np.clip(rad / 900, 0, 1))[..., None]
        return img.astype(np.float32)

    def shot_jars(self, t, u, d, punch):
        # which jar is being talked about: drift the camera towards it
        beat = self.tl.beat_at(t)
        j = beat.get("jar", 1)
        target = self.jars[j]["cx"]
        prev = self.tl.beats[max(0, self.tl.beats.index(beat) - 1)].get("jar", j)
        mix = ease_in_out((t - beat["t0"]) / 0.6)
        fx = (self.jars[prev]["cx"] * (1 - mix) + target * mix) if prev != j else target
        z = (0.98 + 0.03 * ease_in_out(u / d)) * punch
        camx = W / 2 + (fx - W / 2) * 0.08
        camy = 700.0
        # background: the room's window wall, far behind and very soft
        canvas = room_view(self.rooms, (2.0, 330 + (camx - W / 2) * 0.3, 640), 2) * 0.97

        def S(x, y):
            return W / 2 + z * (x - camx), H / 2 + z * (y - camy)
        # table top
        tl_ = S(-200, 900)
        br_ = S(W + 200, 1500)
        cv2.rectangle(canvas, (int(tl_[0]), int(tl_[1])), (int(br_[0]), int(br_[1])), (0.62, 0.39, 0.23), -1)
        edge = S(0, 900)[1]
        canvas[int(max(0, edge)):int(min(H, edge + 6 * z))] *= 0.8
        gy = np.arange(H, dtype=np.float32)
        wood = (1 + 0.04 * np.sin(gy * 0.21) + 0.02 * np.sin(gy * 0.047))[:, None, None]
        canvas[int(max(0, edge)):] *= wood[int(max(0, edge)):]
        state = self.sim.state(u)
        for jj, jar in enumerate(self.jars):
            cx_, top = S(jar["cx"], jar["top"])
            _, bottom = S(jar["cx"], jar["bottom"])
            draw_jar_back(canvas, cx_, top, bottom, (jar["inner_half"] + 14) * z, z)
        # coins (falling ones spin, landed ones lie flat)
        for (x, y, jj, spin, landed) in sorted(state, key=lambda c: c[1]):
            X, Y = S(x, y)
            rx = 27 * z
            if landed is None:
                ry = max(5, abs(math.sin(spin)) * 27) * z
                rxx = rx * (0.75 + 0.25 * abs(math.cos(spin * 0.5)))
            else:
                k = clamp(landed / 0.12)
                ry = (9 + (abs(math.sin(spin)) * 27 - 9) * (1 - k)) * z
                rxx = rx
            coin_draw(canvas, X, Y, rxx, max(3, ry), z)
        for jj, jar in enumerate(self.jars):
            cx_, top = S(jar["cx"], jar["top"])
            _, bottom = S(jar["cx"], jar["bottom"])
            draw_jar_front(canvas, cx_, top, bottom, (jar["inner_half"] + 14) * z, self.jar_colors[jj],
                           self.jar_labels[jj], z)
            n = self.sim.count(jj, u)
            if n:
                last = self.sim.last_landing(jj, u)
                pop = back_out((u - last) / 0.25)
                key = ("cnt", jj, n)
                if key not in self.cast.cache:
                    f = ImageFont.truetype(FONT, 46)
                    txt = f"₹{n * 10}"
                    tw = f.getbbox(txt)[2]
                    im = Image.new("RGBA", (tw + 44, 82), (0, 0, 0, 0))
                    dd = ImageDraw.Draw(im)
                    dd.rounded_rectangle([0, 0, tw + 43, 66], radius=22, fill=(255, 255, 255, 255),
                                         outline=tuple(self.jar_colors[jj]) + (255,), width=5)
                    dd.polygon([(tw / 2 + 10, 64), (tw / 2 + 34, 64), (tw / 2 + 22, 80)], fill=tuple(self.jar_colors[jj]) + (255,))
                    dd.text((22, 4), txt, font=f, fill=(40, 40, 40))
                    self.cast.cache[key] = premul(np.asarray(im))
                img = self.cast.cache[key]
                ih, iw = img.shape[:2]
                blit(canvas, img, affine((iw / 2, ih), (cx_, top - 30 * z), (0.8 + 0.2 * pop) * z * 0.9,
                                         (0.8 + 0.2 * pop) * z * 0.9))
        return canvas

# ------------------------------------------------------------------ sound effects (synthesised)


def synth(kind, rate=AUDIO_RATE, seed=0):
    r = np.random.default_rng(seed)
    if kind in ("coin", "clink"):
        n = int(0.35 * rate)
        tt = np.arange(n) / rate
        f = r.uniform(0.95, 1.08)
        s = (np.sin(2 * np.pi * 2350 * f * tt) * 0.5 + np.sin(2 * np.pi * 3720 * f * tt) * 0.35
             + np.sin(2 * np.pi * 5100 * f * tt) * 0.15) * np.exp(-tt * (14 if kind == "coin" else 22))
        return s * (0.32 if kind == "coin" else 0.16)
    if kind == "whoosh":
        n = int(0.32 * rate)
        noise = r.normal(0, 1, n)
        env = np.sin(np.linspace(0, np.pi, n)) ** 2
        k = np.ones(25) / 25
        return np.convolve(noise, k, "same") * env * 0.10
    if kind == "pop":
        n = int(0.12 * rate)
        tt = np.arange(n) / rate
        return np.sin(2 * np.pi * (500 + 2600 * tt) * tt) * np.exp(-tt * 40) * 0.35
    if kind == "boing":
        n = int(0.45 * rate)
        tt = np.arange(n) / rate
        f = 180 + 140 * np.exp(-tt * 6) * np.cos(2 * np.pi * 9 * tt)
        return np.sin(2 * np.pi * np.cumsum(f) / rate) * np.exp(-tt * 6) * 0.3
    if kind == "step":
        n = int(0.08 * rate)
        tt = np.arange(n) / rate
        return np.convolve(r.normal(0, 1, n), np.ones(40) / 40, "same") * np.exp(-tt * 60) * 0.5
    if kind == "ding":
        n = int(0.9 * rate)
        tt = np.arange(n) / rate
        return (np.sin(2 * np.pi * 1320 * tt) + 0.4 * np.sin(2 * np.pi * 2640 * tt)) * np.exp(-tt * 4) * 0.18
    return np.zeros(1)


def collect_sfx(show):
    tl = show.tl
    ev = list(show.sfx)
    for s in tl.shots:
        if s["shot"] == "wide_run_in":
            ev += [(s["t0"] + k, "step") for k in (0.18, 0.36, 0.54, 0.72, 0.9)]
            ev += [(s["t0"] + 1.15 + 0.3, "boing"), (s["t0"] + 1.55, "pop")]
        if s["shot"] == "wide_end":
            b = s["beats"][0]
            sp = b["speak"]
            ev += [(sp + 0.2 + 0.7 * k + 0.2, "boing") for k in range(2)]
            v0 = sp + b["dur_voice"] - 0.4
            ev += [(v0 + 0.25 * i, "pop") for i in range(3)] + [(v0 + 0.8, "ding")]
        if s["shot"] == "close":
            ev.append((s["t0"] + 0.02, "pop"))
    return ev


def read_wav(path):
    import wave
    with wave.open(str(path), "rb") as w:
        n, ch, sw, rate = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype={2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    a /= 32768.0 if sw == 2 else 2147483648.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if rate != AUDIO_RATE:
        a = np.interp(np.arange(int(len(a) * AUDIO_RATE / rate)) * rate / AUDIO_RATE, np.arange(len(a)), a).astype(np.float32)
    return a


def write_wav(path, a):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def mix_audio(show, voice_dir, path):
    n = int(show.tl.total * AUDIO_RATE) + AUDIO_RATE
    mix = np.zeros(n, np.float32)
    for b in show.tl.beats:
        wav = Path(voice_dir) / f"{b['id']}.wav"
        if wav.exists():
            a = read_wav(wav)
            i = int(b["speak"] * AUDIO_RATE)
            mix[i:i + len(a)] += a[: max(0, n - i)]
    for k, (te, kind) in enumerate(collect_sfx(show)):
        s = synth(kind, seed=k).astype(np.float32)
        i = int(te * AUDIO_RATE)
        if 0 <= i < n:
            mix[i:i + len(s)] += s[: n - i]
    peak = np.abs(mix).max()
    if peak > 0.98:
        mix *= 0.98 / peak
    write_wav(path, mix[: int(show.tl.total * AUDIO_RATE)])

# ------------------------------------------------------------------ main

_SHOW = None


def _render(i):
    f = _SHOW.frame(i / FPS)
    return (f * 255).astype(np.uint8).tobytes()


def main():
    global _SHOW
    args = sys.argv[1:]
    show_dir = args[0]
    voice_dir = Path("out/show")
    voice = {}
    if (voice_dir / "voice.json").exists():
        voice = json.loads((voice_dir / "voice.json").read_text())
    _SHOW = Show(show_dir, voice)
    tl = _SHOW.tl
    name = Path(show_dir).name
    os.makedirs("out/show", exist_ok=True)
    print(f"{name}: {tl.total:.1f} s, {len(tl.shots)} shots, voice={'yes' if voice else 'no (silent draft)'}")
    if "--stills" in args:
        ts = [float(x) for x in args[args.index("--stills") + 1].split(",")]
        for t in ts:
            f = _SHOW.frame(t)
            Image.fromarray((f * 255).astype(np.uint8)).save(f"out/show/still_{t:05.2f}.png")
        print("stills written")
        return
    a, b = 0.0, tl.total
    if "--range" in args:
        i = args.index("--range")
        a, b = float(args[i + 1]), float(args[i + 2])
    frames = range(int(a * FPS), int(b * FPS))
    silent = "out/show/_video.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-pix_fmt", "yuv420p", "-crf", "19", silent], stdin=subprocess.PIPE)
    jobs = int(os.environ.get("RENDER_JOBS", os.cpu_count() or 1))
    with Pool(jobs) as pool:
        for k, buf in enumerate(pool.imap(_render, frames, chunksize=6)):
            p.stdin.write(buf)
            if k % 96 == 0:
                print(f"  frame {k}/{len(frames)}", flush=True)
    p.stdin.close()
    p.wait()
    wav = "out/show/_mix.wav"
    mix_audio(_SHOW, voice_dir, wav)
    out = f"out/show/{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-ss", f"{a}", "-t", f"{b - a}", "-i", wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
