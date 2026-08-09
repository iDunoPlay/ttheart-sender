# Templates

Cropped screenshots of the things you want to click. Reference them from flows
by file name, with or without the extension:

```yaml
- find_click: {template: heart_button}
```

## Creating one

```
python main.py prepare            # park LDPlayer at the top-left first
python main.py snip heart_button  # point at two corners, press F8 each time
python main.py find heart_button  # verify it matches, see the confidence score
```

## Tips

- **Crop tight.** Include the button's distinctive artwork, exclude background
  that changes between runs (counters, timers, avatars).
- **Always re-snip after changing the emulator resolution.** Template matching
  is scale-sensitive; alternatively add `scales: [0.9, 1.0, 1.1]` in
  `config.yaml`.
- **Transparent PNGs work.** Alpha-zero pixels are ignored during matching, so
  you can cut a round button out of a busy background.
- **Check the score before trusting a threshold.** `python main.py find <name>`
  prints the best similarity even when nothing matched.

Subfolders are fine — reference them as `menu/settings_icon`.
