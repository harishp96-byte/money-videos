"""Make one picture with OpenAI from the character sheet + a scene prompt.

Env: OPENAI_API_KEY, SCENE (text), REF (path to reference), OUT_NAME, MODEL (optional)
Writes assets/generated/<OUT_NAME>.png and appends to assets/generated/last_run.txt
(the log is committed so Claude can read what happened).
"""
import base64, os, sys, time, traceback
from pathlib import Path
from openai import OpenAI

out_dir = Path("assets/generated"); out_dir.mkdir(parents=True, exist_ok=True)
log = out_dir / "last_run.txt"
lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')}"]

STYLE = (
    "Keep the exact same boy character as in the reference sheet: same face, hair, skin tone, "
    "outfit, proportions. Modern Indian 2D TV cartoon, thick clean black outlines, bright flat "
    "colours with light soft shading, big round white eyes with solid black pupils and a shine. "
    "Detailed, colourful background. No watermark. "
)
scene = os.environ["SCENE"]
ref = os.environ.get("REF", "assets/characters/boy_v1_sheet.png")
name = os.environ.get("OUT_NAME", "test1")
models = [m for m in [os.environ.get("MODEL"), "gpt-image-2.5-sunburst", "gpt-image-2", "gpt-image-1.5", "gpt-image-1"] if m]

client = OpenAI()
done = False
for m in models:
    try:
        with open(ref, "rb") as f:
            r = client.images.edit(model=m, image=[f], prompt=STYLE + scene, size="1536x1024", n=1)
        (out_dir / f"{name}.png").write_bytes(base64.b64decode(r.data[0].b64_json))
        lines.append(f"OK model={m} -> {name}.png usage={getattr(r, 'usage', None)}")
        done = True
        break
    except Exception as e:
        lines.append(f"FAIL model={m}: {type(e).__name__}: {str(e)[:400]}")
log.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if done else 1)
