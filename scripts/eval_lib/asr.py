"""Whisper transcription (faster-whisper) and character error rate.

Audio is passed as a 16 kHz float32 array, never a path: faster-whisper 1.2.1
path decoding breaks with av 19. Text normalization is NFKC + lowercase, then
every character that is not a letter or digit (punctuation, symbols such as
the music note, and ALL whitespace) is dropped, because Whisper's Korean
spacing is inconsistent.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Sequence

import numpy as np
import torch  # noqa: F401  # load torch's cuDNN DLLs before ctranslate2 on Windows

from . import gpu

MODEL_NAME = "large-v3-turbo"
COMPUTE_TYPE = "float16"
DECODE_OPTIONS: dict[str, Any] = {
    "language": "ko",
    "beam_size": 5,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "vad_filter": False,
}


def cache_tag() -> str:
    """Identifier of model + decoding settings; changes invalidate cached transcripts."""
    spec = json.dumps({"model": MODEL_NAME, "compute_type": COMPUTE_TYPE, **DECODE_OPTIONS}, sort_keys=True)
    return f"{MODEL_NAME}_{hashlib.sha1(spec.encode()).hexdigest()[:8]}"


class Transcriber:
    def __init__(self, device: str) -> None:
        from faster_whisper import WhisperModel

        compute_type = COMPUTE_TYPE if device.startswith("cuda") else "int8"
        self.model = WhisperModel(MODEL_NAME, device="cuda" if device.startswith("cuda") else "cpu", compute_type=compute_type)

    def transcribe(self, wav16k: np.ndarray) -> str:
        segments, _ = self.model.transcribe(np.ascontiguousarray(wav16k, dtype=np.float32), **DECODE_OPTIONS)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def close(self) -> None:
        del self.model
        gpu.release()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in ("L", "N"))


def cer(reference: str, hypothesis: str) -> float | None:
    """CER on normalized text; None when the normalized reference is empty."""
    import jiwer

    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return None
    if not hyp:
        return 1.0
    return float(jiwer.cer(ref, hyp))


def corpus_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float | None:
    """Total character edits / total reference characters over clips with a non-empty reference."""
    import jiwer

    pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(references, hypotheses)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return None
    edits = 0
    total = 0
    for ref, hyp in pairs:
        total += len(ref)
        if not hyp:
            edits += len(ref)
            continue
        out = jiwer.process_characters(ref, hyp)
        edits += out.substitutions + out.deletions + out.insertions
    return edits / total
