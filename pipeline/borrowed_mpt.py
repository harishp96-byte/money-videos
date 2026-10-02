"""Helpers adapted from MoneyPrinterTurbo (MIT licence).

Source: https://github.com/harry0703/MoneyPrinterTurbo  (Copyright (c) 2024 Harry)
Full licence text: THIRD_PARTY_NOTICES.md

What we borrowed, and changed for our Shorts:
  spring_scale        caption "pop" animation      (app/services/video.py)
  cover_box           crop a picture to fill a frame (app/services/video.py)
  zoom_factor         slow Ken Burns zoom on stills  (app/services/video.py)
  split_sentences     sentence splitter, + Hindi "।"  (app/utils/utils.py)
  similarity          Levenshtein text match          (app/services/subtitle.py)
  whisper_cues        Whisper speech -> timed lines   (app/services/subtitle.py)
  align_to_script     fix Whisper text with the script (app/services/subtitle.py)
Not borrowed: its stock-footage downloads, Edge TTS voice (unofficial use of a
Microsoft service) and bundled music/fonts (licences unclear for monetised use).
"""
import math

# ---- caption pop (video.py: _get_subtitle_spring_scale) ---------------------
SPRING_SECONDS = 0.25          # MPT uses 0.18; a touch slower reads better for kids
MIN_SCALE, MAX_SCALE = 0.05, 1.35


def spring_scale(t, duration=SPRING_SECONDS):
    """Scale of a caption t seconds after it appears: overshoots, then settles at 1."""
    if duration <= 0 or t >= duration:
        return 1.0
    p = max(0.0, min(t / duration, 1.0))
    s = 1.0 - math.exp(-6.0 * p) * math.cos(2.5 * math.pi * p)
    return max(MIN_SCALE, min(s, MAX_SCALE))


# ---- fit a picture to a frame (video.py: _fit_clip_to_canvas, "cover") -------
def cover_box(src_w, src_h, dst_w, dst_h):
    """Return (scale, crop_x, crop_y) so the picture fills dst exactly, cropped evenly."""
    scale = max(dst_w / src_w, dst_h / src_h)
    rw, rh = max(dst_w, math.ceil(src_w * scale)), max(dst_h, math.ceil(src_h * scale))
    return scale, (rw - dst_w) // 2, (rh - dst_h) // 2


# ---- slow zoom on still images (video.py: render_image_zoom_video) -----------
def zoom_factor(t, duration, rate=0.03, cap=0.15):
    """MPT zooms about 3% per second. We cap it so long scenes don't over-zoom."""
    return 1.0 + min(rate * duration, cap) * (t / max(duration, 1e-6))


# ---- sentence splitting (utils.py: split_string_by_punctuations) -------------
PUNCTUATIONS = ["?", ",", ".", ";", ":", "!", "…", "।", "॥"]


def split_sentences(s, keep_commas=False):
    """Split text at punctuation. Keeps 2.5 and 10,000 whole, like MPT.
    keep_commas=True gives longer caption lines (split only at . ! ? ।)."""
    stops = [p for p in PUNCTUATIONS if keep_commas is False or p != ","]
    out, cur = [], ""
    for i, ch in enumerate(s):
        prev = s[i - 1] if i > 0 else ""
        nxt = s[i + 1] if i < len(s) - 1 else ""
        if ch == "\n":
            out.append(cur.strip()); cur = ""; continue
        if ch in ".," and prev.isdigit() and nxt.isdigit():
            cur += ch; continue
        if ch in stops:
            cur += ch if ch in "?!।" else ""
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return [x for x in out if x]


# ---- text similarity (subtitle.py) -------------------------------------------
def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a, b):
    m = max(len(a), len(b))
    return 1 - levenshtein(a.lower(), b.lower()) / m if m else 1.0


# ---- Whisper speech -> timed cues (subtitle.py: create) ----------------------
def whisper_cues(audio_file, lang, model_size="small"):
    """List of (start, end, text) per spoken phrase, or None if Whisper isn't installed."""
    try:
        from faster_whisper import WhisperModel  # MIT licence
    except ImportError:
        return None
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        audio_file, language=lang, beam_size=5, word_timestamps=True,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
    )
    cues = []
    for seg in segments:
        words = seg.words or []
        if not words:
            if seg.text.strip():
                cues.append((seg.start, seg.end, seg.text.strip()))
            continue
        start, text = None, ""
        for w in words:
            start = w.start if start is None else start
            text += w.word
            if w.word.rstrip().endswith(tuple(PUNCTUATIONS)):
                cues.append((start, w.end, text.strip()))
                start, text = None, ""
        if text.strip():
            cues.append((start, words[-1].end, text.strip()))
    return cues


# ---- put the script's exact words on Whisper's timings (subtitle.py: correct)
def align_to_script(cues, script_lines):
    """For each script line return (start, end). Whisper's own words are thrown
    away; only its timing is kept, so spelling always matches the script."""
    out, ci = [], 0
    for line in script_lines:
        if ci >= len(cues):
            out.append(None)
            continue
        start, end, heard = cues[ci]
        nxt = ci + 1
        while nxt < len(cues) and similarity(line, heard + " " + cues[nxt][2]) > similarity(line, heard):
            heard += " " + cues[nxt][2]
            end = cues[nxt][1]
            nxt += 1
        out.append((start, end))
        ci = nxt
    return out
