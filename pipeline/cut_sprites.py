"""Cut the boy's poses out of the character sheet into transparent PNGs.

Run once (or whenever the sheet changes):
    python pipeline/cut_sprites.py
Writes assets/characters/sprites/<pose>.png. The white sheet background is
removed by flood-filling from the edges of each box, so white parts inside
the drawing (eyes, teeth, shoes) stay solid.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

SHEET = Path("assets/characters/boy_v1_sheet.png")
OUT = Path("assets/characters/sprites")

# (left, top, right, bottom) on the 1672x941 sheet
BOXES = {
    "stand": (215, 5, 475, 575),      # full body, front, smiling
    "turn": (690, 5, 935, 580),       # full body, three-quarter, smiling
    "happy": (70, 560, 370, 940),     # close-ups (head and shoulders)
    "think": (480, 560, 775, 940),
    "wow": (905, 560, 1210, 940),
    "sad": (1310, 560, 1610, 940),
}


def cut(img):
    a = np.asarray(img.convert("RGB")).astype(int)
    light = (a.min(axis=2) > 228) & ((a.max(axis=2) - a.min(axis=2)) < 22)
    lab, _ = ndimage.label(light)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge))
    bg = ndimage.binary_opening(bg, iterations=1)
    # keep only the biggest drawing in the box (drops crumbs of nearby poses)
    fg_lab, n = ndimage.label(~bg)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(fg_lab), fg_lab, range(1, n + 1))
        bg = fg_lab != (int(np.argmax(sizes)) + 1)
    alpha = Image.fromarray(np.where(bg, 0, 255).astype(np.uint8))
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    out = img.convert("RGBA")
    out.putalpha(alpha)
    return out.crop(out.getbbox())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = Image.open(SHEET)
    for name, box in BOXES.items():
        sprite = cut(sheet.crop(box))
        sprite.save(OUT / f"{name}.png")
        print(name, sprite.size)


if __name__ == "__main__":
    main()
