"""Family asset inventory: what exists, what is missing, and the Gemini prompt for each.

    python pipeline/inventory.py status                 # progress by category
    python pipeline/inventory.py next 5                 # prompts for the next 5 missing items
    python pipeline/inventory.py next 5 --cat places    # only one category
    python pipeline/inventory.py list                   # every id with done/missing
    python pipeline/inventory.py add dad.pose.run file.png    # store a finished file under its right name
    python pipeline/inventory.py add aarav.turnaround f.png --crop 0,0,1200,768   # optional crop l,t,r,b

An item is DONE when its file exists. Definitions live in assets/family/inventory.yaml.
"""
import json
import os
import shutil
import sys

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FAM = os.path.join(ROOT, "assets", "family")
INV = yaml.safe_load(open(os.path.join(FAM, "inventory.yaml"), encoding="utf-8"))
STYLE, NEG = INV["style"], INV["negative"]
PLAIN = "plain soft beige studio background, even lighting"


def items():
    """Yield dicts: id, cat, file (relative to assets/family), prompt (dict)."""
    out = [dict(id="master.family_sheet", cat="master", file="master/family_sheet.jpg", prompt=None)]
    for key, c in INV["characters"].items():
        who = c["who"]
        keep = "exact same face, hair, outfit and look as the attached family sheet"
        out.append(dict(id=f"{key}.sheet", cat="sheets", file=f"{key}/sheet.jpg", prompt={
            "task": "complete character reference sheet, one image", "character": who,
            "keep": keep + ", identical in every cell",
            "render": "3D rendered CGI character, Pixar-style, soft skin shading, realistic hair, NOT 2D, NOT a flat illustration, no outlines",
            "layout": {"top": "full-body turnaround: front, three-quarter left, side left, back (and three-quarter right if it fits), neutral standing pose",
                       "right_top": "head-and-shoulder expressions: " + ", ".join(INV["expressions"]),
                       "bottom": "full-body action poses in one row: walking, running, jumping, sitting on a chair, sitting cross-legged, crouching, pointing, waving"},
            "background": "plain soft beige, even lighting, same scale and lighting in every cell",
            "aspect_ratio": "16:9", "negative": NEG + ", no 2D drawing"}, optional=False))
        out.append(dict(id=f"{key}.turnaround", cat="extras", optional=True, file=f"{key}/turnaround.jpg", prompt={
            "task": "character turnaround sheet, one image", "character": who, "keep": keep,
            "layout": "top row: full-body views in a line, front, three-quarter left, side left, back, three-quarter right. "
                      "Bottom row: head-only views, front, three-quarter, side, back of head. Same scale in every view, "
                      "neutral standing pose, arms slightly away from the body so hands are visible",
            "background": PLAIN, "aspect_ratio": "16:9", "style": STYLE, "negative": NEG}))
        out.append(dict(id=f"{key}.expressions", cat="extras", optional=True, file=f"{key}/expressions.jpg", prompt={
            "task": "expression sheet, one image", "character": who, "keep": keep,
            "layout": "grid of 12 head-and-shoulder expressions: " + ", ".join(INV["expressions"]),
            "background": PLAIN, "aspect_ratio": "16:9", "style": STYLE, "negative": NEG}))
        poses = dict(INV["common_poses"])
        poses.update(c.get("extra_poses", {}))
        for pid, desc in poses.items():
            out.append(dict(id=f"{key}.pose.{pid}", cat="poses", optional=True, file=f"{key}/pose_{pid}.jpg", prompt={
                "task": "single full-body pose image", "character": who, "keep": keep, "pose": desc,
                "background": PLAIN, "aspect_ratio": "9:16", "style": STYLE, "negative": NEG}))
    for gid, desc in INV["groups"].items():
        out.append(dict(id=f"group.{gid}", cat="groups", file=f"groups/{gid}.jpg", prompt={
            "task": "group image", "scene": desc, "keep": "exact same faces, outfits and look as the attached family sheet",
            "background": PLAIN, "aspect_ratio": "9:16", "style": STYLE, "negative": NEG}))
    for pid, desc in INV["places"].items():
        out.append(dict(id=f"place.{pid}", cat="places", file=f"places/{pid}.jpg", prompt={
            "task": "place still, no people", "scene": desc,
            "style": "Pixar-style 3D animation, warm natural light, shallow depth of field, Indian middle-class setting",
            "aspect_ratio": "9:16", "negative": "no text, no signboards, no posters, no logos, no watermark, no people"}))
    for pid, desc in INV["props"].items():
        out.append(dict(id=f"prop.{pid}", cat="props", file=f"props/{pid}.jpg", prompt={
            "task": "single prop image", "object": desc, "background": PLAIN, "aspect_ratio": "1:1",
            "style": "Pixar-style 3D animation, warm natural light",
            "negative": "no text, no logos, no watermark, no readable numbers"}))
    for cid, desc in INV["clips"].items():
        out.append(dict(id=f"clip.{cid}", cat="clips", file=f"clips/{cid}.mp4", prompt={
            "task": "8 second video clip",
            "references": "attached family sheet and one place still; keep faces and clothes identical",
            "action": desc, "style": "Pixar-style 3D animation, warm natural light, shallow depth of field, steady camera, smooth natural movement",
            "format": "vertical 9:16, 8 seconds", "audio": "soft ambient sound only, no speech, no music",
            "negative": "no speech, no dialogue, no subtitles, no text, no logos, no signboards, no watermark, no extra people, no distorted hands"}))
    for it in out:
        it.setdefault("optional", False)
        it["done"] = os.path.exists(os.path.join(FAM, it["file"]))
    return out


def bar(done, total, width=20):
    n = round(width * done / total) if total else 0
    return "#" * n + "-" * (width - n)


def cmd_status(its):
    cats = []
    for it in its:
        if it["cat"] not in cats:
            cats.append(it["cat"])
    for c in cats:
        sub = [i for i in its if i["cat"] == c]
        d = sum(i["done"] for i in sub)
        print(f"{c:8s} [{bar(d, len(sub))}] {d}/{len(sub)}")
    req = [i for i in its if not i["optional"]]
    d = sum(i["done"] for i in req)
    print(f"{'REQUIRED':8s} [{bar(d, len(req))}] {d}/{len(req)}   (extras and poses are optional)")


def cmd_next(its, argv):
    n = int(argv[0]) if argv and argv[0].isdigit() else 5
    cat = argv[argv.index("--cat") + 1] if "--cat" in argv else None
    todo = [i for i in its if not i["done"] and (not cat or i["cat"] == cat) and i["prompt"] and (cat or not i["optional"])]
    for i in todo[:n]:
        print(f"### {i['id']}  ->  assets/family/{i['file']}")
        print(json.dumps(i["prompt"], indent=2, ensure_ascii=False))
        print()
    print(f"({len(todo)} missing{' in ' + cat if cat else ''})")


def cmd_add(its, argv):
    iid, src = argv[0], argv[1]
    it = next((i for i in its if i["id"] == iid), None)
    if not it:
        sys.exit("unknown id: " + iid)
    dst = os.path.join(FAM, it["file"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if dst.endswith(".mp4"):
        shutil.copy(src, dst)
    else:
        from PIL import Image
        im = Image.open(src).convert("RGB")
        if "--crop" in argv:
            l, t, r, b = [int(v) for v in argv[argv.index("--crop") + 1].split(",")]
            im = im.crop((l, t, r, b))
        im.thumbnail((1600, 1600))
        im.save(dst, quality=90)
    print("saved", dst)


def main():
    argv = sys.argv[1:]
    its = items()
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        cmd_status(its)
    elif cmd == "next":
        cmd_next(its, argv[1:])
    elif cmd == "list":
        for i in its:
            print(("DONE   " if i["done"] else "missing"), i["id"])
    elif cmd == "add":
        cmd_add(its, argv[1:])
    else:
        sys.exit(__doc__)


main()
