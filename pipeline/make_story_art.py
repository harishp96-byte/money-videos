"""Make the pictures for a story (character sheet, then one picture per shot) with OpenAI.

Costs money per picture. Run from GitHub Actions -> "Make story art".

Env:
  OPENAI_API_KEY   required
  STORY_DIR        default stories/kayal-pongal-pot (needs prompts.csv)
  PART             "sheet" (character sheet only) or "shots" (default)
  BEATS            for PART=shots: "1-3", "5", "1,4,9" or "all" (default "1")
  QUALITY          low | medium | high   (default low, cheapest)
  MODEL            optional model name to try first

Writes STORY_DIR/character_sheet.png and STORY_DIR/images/<shot id>.png
(e.g. 1.1 -> images/1_1.png). Existing files are skipped, so a run can be
repeated safely. The log STORY_DIR/last_run.txt is committed.
"""
import base64, csv, os, sys, time
from pathlib import Path
from openai import OpenAI

story = Path(os.environ.get("STORY_DIR", "stories/kayal-pongal-pot"))
part = os.environ.get("PART", "shots")
beats_arg = os.environ.get("BEATS", "1").strip().lower()
quality = os.environ.get("QUALITY", "low")
models = [m for m in [os.environ.get("MODEL"), "gpt-image-2.5-sunburst", "gpt-image-2",
                      "gpt-image-1.5", "gpt-image-1"] if m]

SHEET_PROMPT = (
    "Character reference sheet, hand-painted 2D animation-film look, painterly illustration, "
    "semi-realistic, soft watercolor texture. Left: a 10-year-old Tamil village girl with a long "
    "braid with jasmine flowers, yellow pavadai-davani, silver anklets, front view, side view, "
    "and three expressions (smiling, curious, surprised). Right: a 62-year-old Tamil potter, "
    "silver hair, round wire glasses, white veshti, maroon towel on shoulder, clay-stained "
    "hands, front view, side view, and three expressions. Plain cream background, no text."
)

story.mkdir(parents=True, exist_ok=True)
(story / "images").mkdir(exist_ok=True)
log = story / "last_run.txt"
lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')} part={part} beats={beats_arg} quality={quality}"]
client = OpenAI()


def pick_beats(arg, available):
    if arg == "all":
        return sorted(available)
    out = set()
    for piece in arg.split(","):
        piece = piece.strip()
        if "-" in piece:
            a, b = piece.split("-")
            out.update(range(int(a), int(b) + 1))
        elif piece:
            out.add(int(piece))
    return sorted(out & set(available))


def save(result, path):
    path.write_bytes(base64.b64decode(result.data[0].b64_json))


def generate(make_call, path, label):
    for m in models:
        try:
            save(make_call(m), path)
            lines.append(f"OK {label} model={m}")
            return True
        except Exception as e:
            lines.append(f"FAIL {label} model={m}: {type(e).__name__}: {str(e)[:300]}")
    return False


ok = True
sheet = story / "character_sheet.png"
if part == "sheet":
    if sheet.exists():
        lines.append("character_sheet.png already exists, skipped")
    else:
        ok = generate(lambda m: client.images.generate(
            model=m, prompt=SHEET_PROMPT, size="1536x1024", quality=quality, n=1),
            sheet, "character_sheet")
else:
    if not sheet.exists():
        lines.append("STOP: character_sheet.png missing. Run PART=sheet and approve the picture first.")
        ok = False
    else:
        with open(story / "prompts.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_beat = {}
        for r in rows:
            by_beat.setdefault(int(r["id"].split(".")[0]), []).append(r)
        for beat in pick_beats(beats_arg, by_beat):
            for r in by_beat[beat]:
                out = story / "images" / (r["id"].replace(".", "_") + ".png")
                if out.exists():
                    lines.append(f"skip {r['id']} (exists)")
                    continue

                def call(m, prompt=r["prompt"]):
                    with open(sheet, "rb") as ref:
                        return client.images.edit(model=m, image=[ref], prompt=prompt,
                                                  size="1536x1024", quality=quality, n=1)

                if not generate(call, out, r["id"]):
                    ok = False

log.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if ok else 1)
