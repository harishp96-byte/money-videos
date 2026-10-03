# money-videos

Free, automatic video factory for the finance channels (English, Hindi, Tamil).
Write an episode script, push it, and GitHub renders three colourful animated
videos. The Hindi one comes back already voiced.

## Weekly use (phone)
1. Claude adds `episodes/<week>/script.yaml` to this repo. Rendering starts by itself.
2. Open the **Actions** tab → newest **Render episode** run (green tick when done).
3. Scroll to **Artifacts** and download `<week>-en`, `<week>-hi`, `<week>-ta`.
4. Hindi: ready to edit and post. English and Tamil: silent video plus a
   `readalong.txt` with the time each line starts. Record your voice over it in
   Adobe Express or Canva.

To re-run by hand: Actions → Render episode → **Run workflow**.

## Right now: English only, with an AI voice
The render currently makes only the English video, voiced for free by
Kokoro-82M (`pipeline/tts_en.py`). The voice also sets the timing: captions
appear word-group by word-group as they are spoken and the boy's mouth moves
with the loudness of the voice. Change the voice in `script.yaml`
(`voice_en: {voice, speed, pitch}`). Tamil and Hindi come back once English
looks right. The newest videos are also copied to the `previews` branch.

## House style (cartoon, v2: "grounded, full-frame")
`pipeline/cartoon.py` makes every video look like a small Indian 2D cartoon
show, not a slideshow: the boy from `assets/characters/boy_v1_sheet.png` lives
inside drawn rooms, with quick cuts every 1-2 s (title cards, wide shots,
close-up reactions) and light sound effects. Each render gives two files per
language, each composed for its own shape:
- `<week>_<lang>.mp4`: 16:9 long video
- `<week>_<lang>_short.mp4`: 9:16 Short/Reel that fills the whole frame. A
  small "EP n + title" pill sits at the top; captions sit in the lower third,
  inside the Shorts safe area, and move above his face in close-ups.

What keeps the boy from looking like a sticker:
- He stands on the floor: soft contact shadow under his shoes, a long shadow
  falling away from the window, warm room light on him with a rim of window
  light, and his size follows the floor perspective.
- The room is in layers (room, actors, blurred foreground leaves) and the
  camera moves them at different speeds, so it feels 2.5D. Close shots blur
  the room a little.
- He is never frozen: breathing, blinking every few seconds, a sway while he
  talks, step-bob when he walks in, a squash when he lands, mouth flaps.
- Ideas are acted out with props on tables and floors: the salary envelope's
  notes fly into three labelled jars (50/30/20); notes from a ₹30,000 stack
  turn into coins for the piggy bank while the counter rises to ₹6,000; tips
  get their own prop (piggy, phone with an auto-save switch, shopping bag).
- Captions are word-by-word: the word being spoken turns yellow.

Script extras: `ep:` (episode number), `calendar: ["1", "20"]` and
`mood: sad|wow|think|happy` on a title scene, `total: "₹30,000"` on a number
scene (label on the stack of notes), `icons: [piggy, phone, bag]` on a bullets
scene (one prop per tip). Poses are cut from the sheet by
`python pipeline/cut_sprites.py` (run again if the sheet changes; it also
removes the sheet's white gap between his legs).

Render by hand: `python pipeline/cartoon.py episodes/<week> ta` (add
`--preview 10` for the first 10 s, `--only short` for just the Short).
It uses every CPU core (`RENDER_JOBS=2` to limit).

## Older Shorts maker (not run by default)
`pipeline/make_short.py` is the earlier picture-slideshow Short. The render
no longer runs it (the cartoon renderer makes the Short now), but it still
works by hand and supports your own voice recording with Whisper timing. Each has the MP4 and
an `.srt` caption file (upload it in YouTube Studio → Subtitles).
Layout: title band on top, the scene picture slowly zooming or panning,
the spoken line as a popping caption, disclaimer at the bottom.

Optional extras, all added from the phone browser
(open the folder on github.com → **Add file → Upload files** → **Commit**):

| Add this | Where | What happens |
| --- | --- | --- |
| One picture per scene (make them free in the Gemini or ChatGPT app) | `episodes/<week>/images/s1.png`, `s2.png`, … (named after the segment `id`) | Shown in the middle. Missing ones get a colour card. |
| Your voice for a language (phone voice recorder is fine) | `episodes/<week>/voice_en.m4a` or `voice_ta.m4a` | Scenes and captions follow your recording; Whisper finds the timing, the words come from the script. |
| Background music you are licensed to use (e.g. YouTube Audio Library) | `assets/music/` | Played quietly under the voice. |

Uploading into `episodes/` starts a new render by itself.

## Borrowed code
`pipeline/borrowed_mpt.py` adapts helpers from MoneyPrinterTurbo (MIT): caption
pop animation, picture fitting, slow zoom, sentence splitting and Whisper caption
timing. See `THIRD_PARTY_NOTICES.md`. Its music, fonts, stock footage and Edge
TTS voice are not used (unclear for monetised videos).

## One-time setup for the Hindi voice
The voice model (AI4Bharat Indic Parler-TTS, Apache-2.0) is free but needs a
free Hugging Face login.
1. Sign up at huggingface.co, open huggingface.co/ai4bharat/indic-parler-tts and
   tap **Agree and access**.
2. Settings → Access Tokens → **Create new token** → type **Read** → copy it.
3. In this repo: Settings → Secrets and variables → Actions →
   **New repository secret**, name `HF_TOKEN`, paste the token.

Without the token everything still renders; the Hindi video is just silent.

## What's inside
| Path | Does |
| --- | --- |
| `episodes/*/script.yaml` | One episode: scenes, on-screen text and lines in 3 languages |
| `pipeline/tts_hindi.py` | Hindi voice. Change `VOICE` to change the tone |
| `pipeline/cartoon.py` | The cartoon engine: 16:9 video + 9:16 Short |
| `pipeline/cut_sprites.py` | Cuts the boy's poses out of the character sheet and cleans them |
| `pipeline/scene.py` | Old slide-style engine (Manim), no longer used |
| `pipeline/finish.py` | Adds Hindi voice; writes read-along files |
| `pipeline/make_short.py` | Vertical Short/Reel with captions |
| `pipeline/borrowed_mpt.py` | Helpers adapted from MoneyPrinterTurbo (MIT) |
| `.github/workflows/render.yml` | The free cloud render |

Keep this repo **public** so GitHub Actions minutes stay free and unlimited.
All content is for education only, not investment advice.

## 3D prop shots with headless Blender (only where needed)
Character acting comes from the Gemini clips. For a few shots that need real
3D objects (coins dropping into jars, piles of money) GitHub can run Blender
with no computer on your side:
1. Actions -> **Blender prop shot** -> **Run workflow** (tick *quick look* first
   for a single fast still, then run again for the full video).
2. Result: `<shot>_alpha.webm` (transparent, lay it over footage),
   `<shot>_preview.mp4` and a still. They are downloadable as an artifact and
   also on the `previews-props` branch.
3. Script: `pipeline/blender_props.py`. Shot `jars` takes `counts` (coins in
   the spend / save / share jars, e.g. `5 3 2` for 50/30/20). Add new shots in
   the same file. CPU rendering is slow, so shots stay short (about 8 s).
