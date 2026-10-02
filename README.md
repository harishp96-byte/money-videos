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
| `pipeline/scene.py` | The animation engine (Manim) |
| `pipeline/finish.py` | Adds Hindi voice; writes read-along files |
| `.github/workflows/render.yml` | The free cloud render |

Keep this repo **public** so GitHub Actions minutes stay free and unlimited.
All content is for education only, not investment advice.
