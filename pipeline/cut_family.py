"""Cut a character's sheet into transparent sprites.

    python pipeline/cut_family.py aarav

Reads assets/family/<char>/sheet.jpg and sheet_layout.json, removes the plain
beige background of each cell (flood fill that follows the background's soft
gradient, so floor shadows go too) and writes assets/family/<char>/cut/<cell>.png
(RGBA, tightly cropped). The puppet renderer (pipeline/puppet.py) uses these.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def cut_cell(rgb):
    """GrabCut with the cell border as sure background (the cells have plain beige backgrounds)."""
    h, w, _ = rgb.shape
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    ring = np.concatenate([rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3),
                           rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)])
    med = np.median(ring, axis=0)
    m[:4], m[-4:], m[:, :4], m[:, -4:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    near = np.abs(rgb.astype(int) - med).max(axis=2) < 9          # the plain background colour
    m[near & (m == cv2.GC_PR_FGD)] = cv2.GC_PR_BGD
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, m, None, bgm, fgm, 6, cv2.GC_INIT_WITH_MASK)
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg = (lab == keep).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    alpha = cv2.GaussianBlur(cv2.erode(filled, np.ones((2, 2), np.uint8)).astype(np.float32) * 255, (0, 0), 0.9)
    rgba = np.dstack([rgb, alpha.astype(np.uint8)])
    ys, xs = np.where(alpha > 8)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    char = sys.argv[1]
    base = os.path.join(ROOT, "assets", "family", char)
    layout = json.load(open(os.path.join(base, "sheet_layout.json")))
    sp = os.path.join(base, "sheet.jpg")
    if not os.path.exists(sp):
        sp = os.path.join(base, "sheet.png")
    sheet = np.array(Image.open(sp).convert("RGB"))
    out = os.path.join(base, "cut")
    os.makedirs(out, exist_ok=True)
    for name, (l, t, r, b) in layout["cells"].items():
        Image.fromarray(cut_cell(sheet[t:b, l:r])).save(os.path.join(out, name + ".png"))
        print("cut", name)


if __name__ == "__main__":
    main()
