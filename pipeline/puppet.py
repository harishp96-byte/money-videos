"""Puppet animation of a family character from its cut sprites (free, CPU only).

    python pipeline/cut_family.py aarav          # once: sheet -> assets/family/aarav/cut/*.png
    python pipeline/puppet.py aarav demo out/puppet_demo.mp4

What makes it move like a cartoon and not like a slideshow:
  * continuous motion at 24 fps: eased slides, bobbing, sway, breathing,
  * squash and stretch around the feet (anticipation before a jump, squash on landing),
  * pose changes are optical-flow morphs (the in-between frames are computed),
  * a parabola jump with a shadow that shrinks in the air,
  * close-up cut-ins on face sprites with a pop and overshoot.
It is limited by the sprites: one sheet gives one walk pose, so legs do not
cycle. Individual full-resolution poses make it sharper.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
W, H, FPS = 540, 960, 24
FLOOR = 820                      # y of the floor line where the feet stand
BODY_PX = 330                    # on-screen height of a standing character

# ---------------------------------------------------------------- easing

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def back_out(x, k=1.9):
    x = clamp(x)
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def spring(x, amp=1.0, freq=3.0, decay=5.0):
    """Damped wobble that starts at `amp` and settles to 0 (squash on landing)."""
    x = max(0.0, x)
    return amp * math.exp(-decay * x) * math.cos(2 * math.pi * freq * x)

# ---------------------------------------------------------------- sprites

class Sprite:
    """RGBA float sprite (premultiplied), anchored at the bottom centre (the feet)."""

    def __init__(self, path, scale):
        im = Image.open(path).convert("RGBA")
        w, h = im.size
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32) / 255.0
        a[..., :3] *= a[..., 3:4]
        self.px = a
        self.h, self.w = a.shape[:2]


def load_sprites(char):
    base = os.path.join(ROOT, "assets", "family", char, "cut")
    ref = Image.open(os.path.join(base, "pose_walk.png")).size[1]       # a standing-height reference
    k = BODY_PX / ref
    sp = {}
    for f in sorted(os.listdir(base)):
        name = f[:-4]
        s = k
        if name.startswith("turn_"):
            s = k * ref / Image.open(os.path.join(base, f)).size[1]     # same height as the walk pose
        sp[name] = Sprite(os.path.join(base, f), s)
    return sp


def on_canvas(sp, size=(300, 360)):
    """Place a sprite at the bottom centre of a small canvas (for morphing)."""
    cw, ch = size
    out = np.zeros((ch, cw, 4), np.float32)
    x0 = (cw - sp.w) // 2
    y0 = ch - sp.h
    out[y0:y0 + sp.h, x0:x0 + sp.w] = sp.px
    return out


class Morph:
    """Optical-flow in-betweens between two sprites (same canvas, feet aligned)."""

    def __init__(self, a, b):
        self.A, self.B = on_canvas(a), on_canvas(b)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        ga = self._gray(self.A)
        gb = self._gray(self.B)
        self.f_ab = dis.calc(ga, gb, None)       # A -> B
        self.f_ba = dis.calc(gb, ga, None)       # B -> A
        h, w = ga.shape
        self.gx, self.gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    @staticmethod
    def _gray(px):
        a = px[..., 3:4]
        rgb = px[..., :3] + (1 - a) * 0.5
        return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    def at(self, t):
        t = clamp(t)
        mapa = (self.gx + t * self.f_ba[..., 0], self.gy + t * self.f_ba[..., 1])
        mapb = (self.gx + (1 - t) * self.f_ab[..., 0], self.gy + (1 - t) * self.f_ab[..., 1])
        wa = cv2.remap(self.A, mapa[0], mapa[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        wb = cv2.remap(self.B, mapb[0], mapb[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out = (1 - t) * wa + t * wb
        ys, xs = np.where(out[..., 3] > 0.02)
        return out, 300, 360


class Pose:
    """Something drawable: premultiplied RGBA array with its feet at the bottom centre."""

    def __init__(self, px):
        self.px = px
        self.h, self.w = px.shape[:2]

# ---------------------------------------------------------------- drawing

def draw_pose(canvas, px, x, y, sx=1.0, sy=1.0, rot=0.0, alpha=1.0, flip=False):
    """Draw premultiplied RGBA `px` with its bottom centre at (x, y)."""
    h, w = px.shape[:2]
    if flip:
        px = px[:, ::-1]
    ca, sa = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    # map: sprite point (u, v) -> screen, scaling and rotating about the feet (w/2, h)
    m = np.array([[ca * sx, -sa * sy, 0], [sa * sx, ca * sy, 0]], np.float32)
    m[:, 2] = np.array([x, y], np.float32) - m[:, :2] @ np.array([w / 2, h], np.float32)
    warped = cv2.warpAffine(np.ascontiguousarray(px), m, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    warped *= alpha
    canvas[:] = warped + canvas * (1 - warped[..., 3:4])


def background():
    """A simple warm room: wall gradient, floor, window light. (Replace with a place still later.)"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wall = np.dstack([236 - 18 * yy / H, 214 - 24 * yy / H, 186 - 28 * yy / H])
    floor_m = (yy > FLOOR - 40).astype(np.float32)[..., None]
    floor = np.dstack([196 - 40 * (yy - FLOOR) / 140, 150 - 36 * (yy - FLOOR) / 140, 112 - 30 * (yy - FLOOR) / 140])
    img = wall * (1 - floor_m) + floor * floor_m
    # window light patch
    d = ((xx - 380) / 230) ** 2 + ((yy - 300) / 330) ** 2
    img += 34 * np.exp(-d)[..., None] * np.array([1.0, 0.92, 0.7], np.float32)
    # vignette
    v = 1 - 0.28 * (((xx - W / 2) / (W / 1.2)) ** 2 + ((yy - H / 2) / (H / 1.2)) ** 2)
    img *= v[..., None]
    # skirting board line
    img[FLOOR - 42:FLOOR - 36] *= 0.8
    return np.clip(img, 0, 255).astype(np.float32) / 255.0


def shadow(canvas, x, strength=0.38, width=120, lift=0.0):
    layer = np.zeros((H, W), np.float32)
    k = 1.0 - 0.45 * clamp(lift / 180.0)
    cv2.ellipse(layer, (int(x), FLOOR + 4), (int(width * k * 0.5), int(14 * k)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 9) * strength * (1.0 - 0.5 * clamp(lift / 180.0))
    canvas[..., :3] *= (1 - layer[..., None] * 0.9)


def pill(canvas, text, cx, cy, scale=1.0, size=34):
    """Rounded caption bubble that pops in."""
    if scale <= 0.02:
        return
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size * scale))
    im = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    tw = d.textlength(text, font=font)
    pad = int(20 * scale)
    th = int(size * scale * 1.2)
    box = (cx - tw / 2 - pad, cy - th / 2 - pad / 2, cx + tw / 2 + pad, cy + th / 2 + pad / 2)
    d.rounded_rectangle(box, radius=int(26 * scale), fill=(255, 255, 255), outline=(40, 40, 40), width=3)
    d.text((cx - tw / 2, cy - th / 2 - 2), text, font=font, fill=(35, 35, 35))
    canvas[:] = np.asarray(im).astype(np.float32) / 255.0

# ---------------------------------------------------------------- the demo timeline

def demo(sp):
    morphs = {
        ("walk", "stand"): Morph(sp["pose_walk"], sp["turn_3q"]),
        ("stand", "wave"): Morph(sp["turn_3q"], sp["pose_wave"]),
        ("wave", "stand"): Morph(sp["pose_wave"], sp["turn_3q"]),
        ("jump", "stand"): Morph(sp["pose_jump"], sp["turn_3q"]),
        ("stand", "point"): Morph(sp["turn_3q"], sp["pose_point"]),
    }
    bg = background()
    total = 10.0
    n = int(total * FPS)
    heads = {k: sp[k] for k in ("ex_happy", "ex_surprised", "ex_thinking")}
    cx = W / 2

    def pose_at(t):
        """Return (px array, x, lift, sx, sy, rot) for the wide shot at time t."""
        # --- walk in
        if t < 2.2:
            u = ease_out(t / 2.2)
            x = -90 + (cx - (-90)) * u
            bob = abs(math.sin(2 * math.pi * 2.3 * t)) * 12 * (1 - 0.6 * u)
            rot = 2.2 * math.sin(2 * math.pi * 2.3 * t)
            sy = 1 + 0.025 * math.sin(2 * math.pi * 4.6 * t)
            return sp["pose_walk"].px, x, bob, 1 / sy, sy, rot
        # --- walk -> stand
        if t < 2.6:
            m, = (morphs[("walk", "stand")].at(smooth((t - 2.2) / 0.4))[0],)
            return m, cx, 0, 1, 1, 0
        # --- idle breathing
        if t < 3.4:
            br = math.sin(2 * math.pi * 0.9 * (t - 2.6))
            return on_canvas(sp["turn_3q"]), cx, 0, 1 - 0.006 * br, 1 + 0.012 * br, 0.8 * br
        # --- wave
        if t < 3.8:
            return morphs[("stand", "wave")].at(smooth((t - 3.4) / 0.4))[0], cx, 0, 1, 1, 0
        if t < 5.0:
            w = math.sin(2 * math.pi * 1.4 * (t - 3.8))
            return on_canvas(sp["pose_wave"]), cx, abs(w) * 6, 1, 1 + 0.01 * w, 2.5 * w
        if t < 5.4:
            return morphs[("wave", "stand")].at(smooth((t - 5.0) / 0.4))[0], cx, 0, 1, 1, 0
        # --- jump: anticipation, take-off, air, landing
        if t < 5.8:                               # anticipation: squash the stand, then hold the crouch
            if t < 5.55:
                u = smooth((t - 5.4) / 0.15)
                return on_canvas(sp["turn_3q"]), cx, 0, 1 + 0.06 * u, 1 - 0.08 * u, 0
            u = smooth((t - 5.55) / 0.25)
            return on_canvas(sp["pose_crouch"]), cx, 0, 1 + 0.03 * u, 1 - 0.05 * u, 0
        if t < 6.0:                               # push off: stretch up
            u = (t - 5.8) / 0.2
            return on_canvas(sp["pose_jump"]), cx, 0, 1 - 0.08 * u, 1 + 0.14 * u, 0
        if t < 6.75:                              # air time, parabola
            u = (t - 6.0) / 0.75
            lift = 4 * u * (1 - u) * 190
            return on_canvas(sp["pose_jump"]), cx, lift, 0.96, 1.06, -3 * (u - 0.5)
        if t < 7.3:                               # landing squash that springs back
            u = (t - 6.75)
            sq = spring(u, 0.16, 2.2, 6.0)
            m = morphs[("jump", "stand")].at(smooth(u / 0.3))[0]
            return m, cx, 0, 1 + sq * 0.9, 1 - sq, 0
        return on_canvas(sp["turn_3q"]), cx, 0, 1, 1, 0

    out = []
    for i in range(n):
        t = i / FPS
        canvas = bg.copy()
        canvas4 = np.dstack([canvas, np.ones((H, W), np.float32)])
        # close-up cut-in (7.4 - 8.9): face with pop and overshoot
        if 7.4 <= t < 8.9:
            blur = cv2.GaussianBlur(bg, (0, 0), 6)
            c4 = np.dstack([blur * 0.92, np.ones((H, W), np.float32)])
            which = "ex_happy" if t < 7.95 else ("ex_surprised" if t < 8.4 else "ex_thinking")
            t0 = {"ex_happy": 7.4, "ex_surprised": 7.95, "ex_thinking": 8.4}[which]
            pop = back_out((t - t0) / 0.25)
            base = 2.9 + 0.02 * math.sin(2 * math.pi * 0.8 * t)
            hs = heads[which]
            # shirt shoulders under the head so it reads as a bust, not a floating head
            sa = np.zeros((H, W), np.float32)
            wd = 215 * pop
            pts = np.array([[cx - 52, 835], [cx - 50, 872], [cx - wd * 0.62, 884], [cx - wd, 925], [cx - wd * 1.05, H],
                            [cx + wd * 1.05, H], [cx + wd, 925], [cx + wd * 0.62, 884], [cx + 50, 872], [cx + 52, 835]], np.int32)
            cv2.fillPoly(sa, [pts], 1.0, cv2.LINE_AA)
            sa = cv2.GaussianBlur(sa, (0, 0), 7)
            grad = np.linspace(0.97, 0.86, H, dtype=np.float32)[:, None]
            shirt = np.dstack([grad, grad, grad * 0.99])
            c4[..., :3] = shirt * sa[..., None] + c4[..., :3] * (1 - sa[..., None])
            sc = base * (0.82 + 0.18 * pop)
            tilt = 2.0 * math.sin(2 * math.pi * 0.6 * t)
            draw_pose(c4, hs.px, cx, 840, sc, sc, tilt)
            frame = c4[..., :3]
        else:
            px, x, lift, sx, sy, rot = pose_at(t)
            shadow(canvas4[..., :3], x, 0.45, 150, lift)
            if px.shape[0] == 360 and px.shape[1] == 300:       # canvas-sized (morph) sprite
                sc = 1.0
            else:
                sc = 1.0
            draw_pose(canvas4, px, x, FLOOR - lift, sx * sc, sy * sc, rot)
            frame = canvas4[..., :3]
            # caption bubble near the end, with a pop
            if t >= 9.0:
                pass
        # last beat: point pose + caption
        if t >= 8.9:
            canvas = bg.copy()
            c4 = np.dstack([canvas, np.ones((H, W), np.float32)])
            u = (t - 8.9) / 0.4
            px = morphs[("stand", "point")].at(smooth(u))[0]
            shadow(c4[..., :3], cx, 0.45, 150, 0)
            draw_pose(c4, px, cx, FLOOR, 1, 1, 0)
            frame = c4[..., :3].copy()
            pill(frame, "Save first, spend later!", cx, 250, back_out((t - 9.2) / 0.3))
        out.append(frame)
    return out


def write_video(frames, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                          "-crf", "18", path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write((np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()


def main():
    char, what = sys.argv[1], sys.argv[2]
    path = sys.argv[3] if len(sys.argv) > 3 else "out/puppet_demo.mp4"
    sp = load_sprites(char)
    frames = demo(sp) if what == "demo" else sys.exit("unknown scene")
    write_video(frames, path)
    print("wrote", path, len(frames), "frames")


if __name__ == "__main__":
    main()
