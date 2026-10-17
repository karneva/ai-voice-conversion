"""Metric modules used by scripts/evaluate.py.

Each GPU-backed module exposes a small class with a ``close()`` method so the
caller can load one model at a time and free it before the next metric runs.

- audio:   loading, 16 kHz resampling, energy-based active-region helpers
- speaker: ECAPA-TDNN speaker embeddings and cosine helpers
- pitch:   torchcrepe F0 extraction and F0 error metrics
- asr:     faster-whisper transcription, text normalization, CER
- mos:     UTMOS22 / SingMOS-Pro naturalness predictors (optional)
- mcd:     mel-cepstral distortion for paired (held-out) clips
- cache:   tiny JSON cache keyed by audio file hash
- gpu:     device selection, model release and per-stage timing/VRAM logging
"""
