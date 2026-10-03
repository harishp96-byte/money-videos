# Family asset inventory

The one-time library for every future episode: the Aarav / Dad / Mom / Dadaji
family sheet, per-character turnarounds, expression sheets and poses, group
shots, 10 places, props and 30 silent Veo clips. The full list and the
outfits are in `inventory.yaml`; a file's presence means the item is done.

```
python pipeline/inventory.py status           # progress bars
python pipeline/inventory.py next 5           # Gemini prompts for the next 5 missing items
python pipeline/inventory.py next 5 --cat poses
python pipeline/inventory.py add dad.pose.run file.png     # saves under the right name
```

## From your phone
1. Ask Claude for the next batch; it prints the prompts (one per missing item).
2. Run them in Gemini with the family sheet attached; download the results.
3. Send the images/clips to Claude in the same order. Claude names, shrinks and
   commits them. Nothing needs renaming by hand.

Images are stored as JPEG (max 1600 px). Clips are stored as .mp4 (about
3-6 MB each). Check the Gemini watermark and commercial-use terms before any
of these appear in a published video.
