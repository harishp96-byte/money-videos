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


def clean_ground(sprite):
    """Full-body poses: drop the sheet's white gap between the legs and the grey
    floor smudge under the shoes. They looked like a paper cut-out on screen.
    The renderer draws its own floor shadow instead."""
    a = np.asarray(sprite).astype(int)
    rgb, al = a[..., :3], a[..., 3]
    h, w = al.shape
    light = (rgb.min(axis=2) > 170) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 35) & (al > 0)
    lab, _ = ndimage.label(light)
    drop = np.zeros_like(light)
    for i, sl in enumerate(ndimage.find_objects(lab)):
        comp = lab == i + 1
        ys, xs = np.nonzero(comp)
        if len(ys) < 60:
            continue
        if ys.mean() > h * 0.7 and abs(xs.mean() - w / 2) < w * 0.17:   # between the legs
            drop |= comp
        elif ys.min() > h - 25:                                          # floor smudge
            drop |= comp
    drop = ndimage.binary_dilation(drop, iterations=1)
    new_a = np.where(drop, 0, al).astype(np.uint8)
    out = sprite.copy()
    out.putalpha(Image.fromarray(new_a).filter(ImageFilter.GaussianBlur(0.6)))
    return out.crop(out.getbbox())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = Image.open(SHEET)
    for name, box in BOXES.items():
        sprite = cut(sheet.crop(box))
        if name in ("stand", "turn"):
            sprite = clean_ground(sprite)
        sprite.save(OUT / f"{name}.png")
        print(name, sprite.size)


if __name__ == "__main__":
    main()
