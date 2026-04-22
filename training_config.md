# RVC training config

## Dataset

- Speaker name: `kyuhyun`
- Raw audio path: `dataset/kyuhyun/raw`
- Processed audio path: `dataset/kyuhyun/processed`
- Target amount: around 20 minutes of clean vocal-only audio
- Slice length: 8 seconds by default, acceptable range 5 to 10 seconds
- Format: mono WAV, 40 kHz sample rate

Use clean, dry vocal material when possible. Avoid heavy reverb, crowd noise, backing vocals, overlapping speakers, clipping, and instrumental bleed.

## Preprocessing

Run from the workspace root:

```bash
python scripts/preprocess_audio.py \
  --input dataset/kyuhyun/raw \
  --output dataset/kyuhyun/processed \
  --sample-rate 40000 \
  --segment-seconds 8
```

Optional flags:

- `--min-segment-seconds 3`: drop very short tail segments.
- `--overwrite`: regenerate files that already exist.
- `--extensions .wav .mp3 .flac .m4a .ogg`: control accepted input types.

## Recommended RVC Settings

| Setting | Value |
| --- | --- |
| Sample rate | 40k |
| F0 method | Harvest |
| Speaker ID | 0 |
| Epoch smoke test | 5 |
| First quality checkpoint | 50 |
| Full training target | 200 |
| Save frequency | 10 epochs |
| Batch size | Start with 4, increase only if VRAM allows |
| Pitch guidance | Enabled for singing voice |

## Runbook

1. Confirm GPU availability in the RVC environment.
2. Preprocess source vocals into `dataset/kyuhyun/processed`.
3. Start a 5 epoch smoke test and confirm logs/checkpoints are created.
4. Train to 50 epochs and run a short inference test.
5. Continue to 200 epochs only if the 50 epoch result is usable.
6. Export both `.pth` and `.index` files into `weights/`.

## Quality Checks

- Pronunciation remains intelligible.
- Pitch follows the source without obvious jumps.
- No frequent robotic buzzing or metallic artifacts.
- No severe clipping in generated audio.
- Timbre similarity improves from 50 to later checkpoints.

## Safety And Rights

Train and use voice models only with appropriate rights and permission. Keep generated files private unless you have clear authorization to publish them.
