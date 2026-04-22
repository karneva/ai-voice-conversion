# RVC AI voice conversion implementation plan

This workspace is prepared for an RVC-based voice conversion experiment. Use only audio you have the right to process, train on, and generate from. Do not publish or distribute a model or generated audio that imitates a real person without the required permission.

## Target Layout

```text
ai-voice-conversion/
├── RVC-WebUI/                # External RVC repository, cloned during setup
├── dataset/
│   └── singer-name/
│       ├── raw/              # Put source vocal audio here
│       └── processed/        # Generated 40 kHz sliced WAV files
├── scripts/
│   └── preprocess_audio.py   # Audio conversion and slicing utility
├── weights/                  # Final .pth and .index exports
└── training_config.md        # Recommended RVC settings and runbook
```

## Prerequisites

- NVIDIA GPU with a working CUDA-compatible PyTorch install.
- Python environment supported by the RVC-WebUI version you use.
- `ffmpeg` and `ffprobe` available on `PATH`.
- Source audio placed in `dataset/singer-name/raw`.

## Setup Steps

1. Clone RVC-WebUI:

   ```bash
   git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git RVC-WebUI
   ```

2. Create and activate a virtual environment inside the RVC project:

   ```bash
   cd RVC-WebUI
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies using the RVC-WebUI documentation for your CUDA and PyTorch version.

4. Add vocal-only source files to:

   ```text
   dataset/singer-name/raw/
   ```

5. Preprocess the dataset:

   ```bash
   python scripts/preprocess_audio.py \
     --input dataset/singer-name/raw \
     --output dataset/singer-name/processed \
     --sample-rate 40000 \
     --segment-seconds 8
   ```

6. In RVC-WebUI, train using the processed dataset path:

   ```text
   ../dataset/singer-name/processed
   ```

## Training Strategy

1. Run a short smoke test first, around 5 epochs, to verify CUDA, logs, and dataset loading.
2. Run a quality checkpoint at around 50 epochs.
3. Continue to around 200 epochs only if the 50 epoch result has stable pronunciation and pitch tracking.
4. Use Harvest F0 extraction for vocal stability.
5. Test inference with a short 5 to 10 second clip before longer conversions.

## Validation Checklist

- `ffmpeg -version` works.
- `python scripts/preprocess_audio.py --help` works.
- Processed files are mono 40 kHz WAV files.
- RVC-WebUI can load the processed dataset.
- A 5 epoch smoke test completes without CUDA errors.
- A short inference test does not contain severe pitch drift, clipping, or timing artifacts.
