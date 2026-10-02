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

## House style (cartoon, like the reference Short)
`pipeline/cartoon.py` makes every video look like the kids' cartoon Short we
picked as the reference: the boy from `assets/characters/boy_v1_sheet.png`
in drawn rooms, quick cuts every 1-2 s (title cards, wide shots, close-up
reactions), moving props (cash, calendar, coin jars, piggy bank, chalkboard),
mouth flaps while he talks, short all-caps captions and light sound effects.
Each render gives two files per language:
- `<week>_<lang>.mp4`: 16:9 long video, captions at the bottom
- `<week>_<lang>_short.mp4`: 9:16 Short/Reel, with a black frame, a yellow
  "EP n: title" line, the cartoon panel in the middle and the caption under it

Script extras: `ep:` (episode number), `calendar: ["1", "20"]` and
`mood: sad|wow|think|happy` on a title scene. Poses are cut from the sheet
by `python pipeline/cut_sprites.py` (run again if the sheet changes).

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
| `pipeline/cut_sprites.py` | Cuts the boy's poses out of the character sheet |
| `pipeline/scene.py` | Old slide-style engine (Manim), no longer used |
| `pipeline/finish.py` | Adds Hindi voice; writes read-along files |
| `pipeline/make_short.py` | Vertical Short/Reel with captions |
| `pipeline/borrowed_mpt.py` | Helpers adapted from MoneyPrinterTurbo (MIT) |
| `.github/workflows/render.yml` | The free cloud render |

Keep this repo **public** so GitHub Actions minutes stay free and unlimited.
All content is for education only, not investment advice.
