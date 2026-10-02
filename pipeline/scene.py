"""Colourful 2D explainer, driven entirely by episodes/<ep>/script.yaml.

Environment:
  EPISODE_DIR  folder holding script.yaml
  LANG_CODE    en | hi | ta  (which language's on-screen text to use)
  DURATIONS    optional path to durations.json from the Hindi voice step

All three language versions use the same timing, so one script makes three
videos that line up scene for scene.
"""
import json
import os
from pathlib import Path

import yaml
from manim import (
    DOWN, LEFT, RIGHT, UP, Circle, Create, Dot, FadeIn, FadeOut,
    GrowFromCenter, LaggedStart, Rectangle, RoundedRectangle, Scene, Text,
    Transform, VGroup, Write, config, there_and_back,
)

EP = Path(os.environ.get("EPISODE_DIR", "episodes/week00-test"))
LANG = os.environ.get("LANG_CODE", "en")
SCRIPT = yaml.safe_load((EP / "script.yaml").read_text(encoding="utf-8"))
DUR = {}
if os.environ.get("DURATIONS") and Path(os.environ["DURATIONS"]).exists():
    DUR = json.loads(Path(os.environ["DURATIONS"]).read_text())

FONT = {"en": "Noto Sans", "hi": "Noto Sans Devanagari", "ta": "Noto Sans Tamil"}[LANG]

# Bright, saturated palette on a deep background: the "mesmerising" look.
BG = "#1B1036"
INK = "#FFFFFF"
NEED, WANT, SAVE, POP, GOLD = "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#FFC23C"
config.background_color = BG


def t(field):
    """Pick this language's text from a {en, hi, ta} field."""
    return field.get(LANG) or field.get("en") if isinstance(field, dict) else field


def txt(s, size=48, color=INK, weight="BOLD"):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def wrap(s, width=40):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


def coin(r=0.6):
    """The recurring mascot: a smiling gold rupee coin."""
    body = Circle(radius=r, color=GOLD, fill_color=GOLD, fill_opacity=1, stroke_width=6)
    rim = Circle(radius=r * 0.8, color="#E0A100", stroke_width=4)
    sign = Text("₹", font="Noto Sans", font_size=int(r * 70), color="#7A4B00", weight="BOLD")
    eyes = VGroup(Dot(color="#3B2200", radius=r * 0.07).shift(LEFT * r * 0.3 + UP * r * 0.45),
                  Dot(color="#3B2200", radius=r * 0.07).shift(RIGHT * r * 0.3 + UP * r * 0.45))
    return VGroup(body, rim, sign, eyes)


class Episode(Scene):
    def seg_time(self, seg):
        spoken = DUR.get(seg["id"], 0)
        return max(float(seg.get("seconds", 5)), spoken + 0.6 if spoken else 0)

    def run(self, *anims, run_time=1.0, **kw):
        self.play(*anims, run_time=run_time, **kw)
        self.used += run_time

    def caption(self, seg):
        if not SCRIPT.get("captions") or "say" not in seg:
            return None
        cap = txt(wrap(t(seg["say"]), 46), size=26, weight="NORMAL")
        box = RoundedRectangle(corner_radius=0.2, width=cap.width + 0.6, height=cap.height + 0.4,
                               fill_color="#000000", fill_opacity=0.45, stroke_width=0)
        group = VGroup(box, cap).to_edge(DOWN, buff=0.3)
        self.add(group)
        return group

    def construct(self):
        for seg in SCRIPT["segments"]:
            self.used = 0.0
            total = self.seg_time(seg)
            cap = self.caption(seg)
            shown = getattr(self, f"v_{seg['visual']}")(seg)
            left = total - self.used - 0.5
            if left > 0:
                self.wait(left)
            self.play(*[FadeOut(m) for m in [shown, cap] if m is not None], run_time=0.5)

    # ---- visuals -------------------------------------------------------
    def v_title(self, seg):
        m = coin(0.9).shift(UP * 1.2)
        head = txt(t(seg["heading"]), 60).next_to(m, DOWN, buff=0.6)
        head.set_color_by_gradient(WANT, NEED)
        self.run(GrowFromCenter(m), run_time=0.8)
        self.run(Write(head), run_time=1.2)
        self.run(m.animate.shift(UP * 0.35), rate_func=there_and_back, run_time=0.8)
        return VGroup(m, head)

    def v_split(self, seg):
        W, H = 11.0, 1.3
        whole = Rectangle(width=W, height=H, fill_color=POP, fill_opacity=1, stroke_width=0).shift(UP * 0.6)
        full = txt("100%", 44).move_to(whole)
        self.run(GrowFromCenter(whole), FadeIn(full), run_time=1.0)
        cols, parts, labels = [NEED, WANT, SAVE], seg["parts"], t(seg["labels"])
        bars, x = VGroup(), -W / 2
        for p, c in zip(parts, cols):
            w = W * p / 100
            bars.add(Rectangle(width=w - 0.08, height=H, fill_color=c, fill_opacity=1,
                               stroke_width=0).move_to([x + w / 2, whole.get_center()[1], 0]))
            x += w
        self.run(Transform(whole, bars), FadeOut(full), run_time=1.2)
        tags = VGroup()
        for bar, p, lab, c in zip(bars, parts, labels, cols):
            tags.add(VGroup(txt(f"{p}%", 40, color=BG).move_to(bar),
                            txt(lab, 34, color=c).next_to(bar, UP, buff=0.3)))
        self.run(LaggedStart(*[FadeIn(g, shift=UP * 0.3) for g in tags], lag_ratio=0.35), run_time=1.5)
        return VGroup(whole, tags)

    def v_number(self, seg):
        big = txt(seg["value"], 120, color=SAVE).shift(UP * 0.7)
        sub = txt(t(seg["caption"]), 36, weight="NORMAL").next_to(big, DOWN, buff=0.4)
        coins = VGroup(*[coin(0.25).move_to([x, 2.4, 0]) for x in (-4.5, -2.5, 2.5, 4.5)])
        self.run(GrowFromCenter(big), run_time=0.9)
        self.run(FadeIn(sub, shift=UP * 0.2), run_time=0.6)
        self.run(LaggedStart(*[FadeIn(c, shift=DOWN * 1.2) for c in coins], lag_ratio=0.2), run_time=1.2)
        return VGroup(big, sub, coins)

    def v_bullets(self, seg):
        cols = [POP, WANT, SAVE, NEED]
        rows = VGroup()
        for i, item in enumerate(t(seg["items"])):
            rows.add(VGroup(Dot(radius=0.14, color=cols[i % 4]), txt(item, 40)).arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to(UP * 0.6)
        self.run(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.5) for r in rows], lag_ratio=0.6), run_time=2.4)
        return rows

    def v_outro(self, seg):
        m = coin(0.8).shift(UP * 1.6)
        pill = RoundedRectangle(corner_radius=0.35, width=4.2, height=0.8, fill_color=NEED,
                                fill_opacity=1, stroke_width=0).next_to(m, DOWN, buff=0.5)
        sub = txt("SUBSCRIBE", 34).move_to(pill)
        note = txt(t(seg["heading"]), 28, weight="NORMAL").next_to(pill, DOWN, buff=0.5)
        self.run(GrowFromCenter(m), run_time=0.7)
        self.run(Create(pill), FadeIn(sub), run_time=0.8)
        self.run(FadeIn(note), run_time=0.6)
        return VGroup(m, pill, sub, note)
