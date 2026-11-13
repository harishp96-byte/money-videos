# Tamil Village Tales: Story 1, "Kayal and the Pongal Pot"

Tamil-narrated, 5-minute tale of how a village potter makes the clay pot for Pongal.
Planning doc (with review suggestions): https://claude.ai/artifact/TPjHDPQbprY6JyuM3vvgKp

## Channel plan
- Narration: Tamil voice-over recorded by the creator, English captions on screen.
- Setting: Tamil Nadu village crafts and festivals (pongal, kolam, pottery, palm-leaf weaving, fishing nets).
- Format: one craft or festival per story, a step-by-step process, so viewers learn something.
- Length: 4-6 minutes, 13 beats x 5 shots = 65 shots, about 4-5 s each.
- Cast: the same two characters in every story.
- Decide early whether the channel is "made for kids" (no shopping on made-for-kids channels).
- Second channel (no narration): wait until this one has 3 uploads; same visual style, different stories.

## Characters (fixed details, never change)
- Kayal, girl, 10: round face, bright curious eyes, long single braid with jasmine flowers, yellow pavadai-davani, small silver anklets.
- Muthu Thatha, grandfather, 62, potter: silver hair, deep smile lines, round wire glasses, plain white veshti, maroon towel on shoulder, clay-stained hands.

## Style
Character sheet first, then attach it to every shot prompt.
Master style line (suggested wording, replaces "Ghibli-inspired" in the doc):
`Hand-painted 2D animation-film look, painterly illustration, semi-realistic, warm golden-hour light, rural Tamil Nadu village, soft watercolor texture, 16:9, no text, same characters as the reference image`

## Beats, Tamil voice-over (draft, creator to adjust) and English captions
| # | Scene | Tamil | English |
|---|---|---|---|
| 1 | Wide: sunrise village street, Kayal running | பொங்கலுக்கு இன்னும் ஒரு வாரம்தான். | Only a week left for Pongal. |
| 2 | Medium: Thatha at workshop door, Kayal looking up | தாத்தா, இந்த வருடம் எனக்கும் பானை செய்யக் கற்றுக்கொடுங்கள்! | Thatha, teach me to make a pot this year! |
| 3 | Close: hands digging red clay at pond bank | முதலில், குளக்கரையில் இருந்து நல்ல களிமண் எடுக்க வேண்டும். | First, good clay is dug from the pond bank. |
| 4 | Close: Kayal sifting clay | கல்லும் குச்சியும் நீக்கி, மண்ணை சலிக்க வேண்டும். | Stones and sticks are removed and the clay is sifted. |
| 5 | Close: bare feet kneading clay | தண்ணீர் சேர்த்து, காலால் நன்றாக மிதித்து பிசைய வேண்டும். | Water is added and the clay is trodden until smooth. |
| 6 | Medium: Thatha at the wheel | சக்கரம் சுழல, களிமண் மெல்ல மெல்ல பானையாக உயர்கிறது. | As the wheel turns, the clay slowly rises into a pot. |
| 7 | Close: Kayal's first wobbly pot | கயலின் முதல் பானை கொஞ்சம் கோணலாக இருந்தது. | Kayal's first pot came out a little crooked. |
| 8 | Wide: pots drying in the yard | பானைகளை இரண்டு நாள் நிழலிலும் வெயிலிலும் உலர்த்த வேண்டும். | The pots dry for two days, in shade and then sun. |
| 9 | Medium: kiln at dusk | பிறகு, சூளையில் வைத்து தீயில் சுட்டுவிட வேண்டும். | Then they are fired in the kiln. |
| 10 | Close: Thatha taps a baked pot | தட்டினால் டங் என்று ஒலித்தால், பானை தயார். | If it rings when tapped, the pot is ready. |
| 11 | Medium: Kayal decorating the pot | மஞ்சள், குங்குமம் பூசி, கோலம் வரைந்து அலங்கரிக்கிறோம். | It is decorated with turmeric, kumkum and kolam. |
| 12 | Wide: Pongal morning, milk boiling over | பால் பொங்கியதும் எல்லோரும் கூவினர்: பொங்கலோ பொங்கல்! | The milk boiled over and everyone cheered: Pongalo Pongal! |
| 13 | Close: crooked pot beside new ones | அந்த கோணல் பானைதான் தாத்தாவுக்கு மிகவும் பிடித்தது. | That crooked pot was Thatha's favourite. |

Each line covers about 5 s but a beat runs 25-30 s: write 3-4 short lines per beat or leave some shots to ambient sound.
End card: five words with English meanings, said aloud: களிமண் (clay), பானை (pot), சக்கரம் (wheel), சூளை (kiln), பொங்கல்.

## Files
- `prompts.csv`: 65 merged prompts (id, beat, prompt), scene + fixed character details + style line.
- `pipeline/make_story_art.py` and `.github/workflows/make-story-art.yml`: Actions -> "Make story art".
  Step 1 `part=sheet` makes the character sheet (one picture). Look at it. Step 2 `part=shots`, a few beats at a time.
  COSTS MONEY per picture through OpenAI (needs the OPENAI_API_KEY secret); start with quality `low`. Free alternative: Google Flow / Vids image-to-video (see the project's breakdown-to-animation-prompt-kit).

## Phone workflow
1. Character sheet in an image app, keep the best.
2. Each shot with the sheet attached plus the fixed details as text; reject any shot where clothes or face drift.
3. Record the Tamil voice-over one beat at a time (CapCut or voice recorder).
4. CapCut: each still 4-6 s with slow zoom or pan; English captions; soft ambient track (royalty-free); channel name left edge, round avatar bottom right.

## Before every upload
- Story, script and Tamil lines are original, not copied from another channel.
- AI disclosure on in YouTube Studio if images look realistic.
- Title, thumbnail, description specific to this story.
- Same cast, new story and craft each time.
- "Made for kids" setting chosen on purpose.
- Check Partner Program thresholds on YouTube's own help page (rules reported to change from 1 Feb 2027, unverified).
