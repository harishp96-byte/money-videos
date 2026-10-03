# Writing a puppet show (show.yaml)

Each beat is one spoken line. `who` speaks, `say` is the line, `shot` picks the camera.

## Shot types
- `close` – talking close-up. Needs `expression` (happy, curious, …; only faces with landmarks in `FACES` work).
- `jars` – the coins-into-jars table. Needs `jar` (0, 1, 2) and `coins`.
- `stage` – full-body shot built from data (below). Consecutive stage beats are one continuous shot; add `cut: true` to cut.
- `wide_run_in`, `wide_end` – the fixed opening and ending shots of ep01.

## Stage beats
```yaml
- id: b7
  shot: stage
  who: dad
  say: "Saving makes dreams come true."
  props: [jars_full]          # jars_full (default), jars (empty), or [] for none
  cast:
    aarav: {pose: standing, position: left, expression: curious, action: listen}
    dad:
      position: right
      expression: happy
      pose: [standing, point@0.2, wave@2.6]   # pose@seconds after the line starts
```

| Field | Words you can use |
|---|---|
| `pose` | standing, front, side, back, point, wave, crouch, sit, sit_floor, walk, run, jump (or a sprite file name) |
| `position` | left, center, right, or a number (0–900) |
| `expression` | happy, excited, laugh, curious, thinking, surprised, sad, worried, stern, calm, shy, sleepy |
| `action` | idle, talk, listen (nods), enter (walks in), exit (walks out), jump |
| `face` | left / right (optional; by default each character looks at the other) |

In full-body shots the face is too small to swap, so `expression` changes the body acting
(head tilt, head drop, lean, bounce). In `close` shots it picks the face sprite.

Every character gets a soft floor shadow, a tight contact shadow under the feet, a
long cast shadow, breathing, head sway, talk nods and a squash-and-overshoot "pop"
on every pose change.
