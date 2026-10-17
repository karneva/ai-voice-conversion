"""Audio loading and active-region helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16000
FRAME_LENGTH = 640  # 40 ms at 16 kHz
HOP_LENGTH = 160  # 10 ms at 16 kHz, same hop as the pitch tracker
ACTIVE_TOP_DB = 40.0
ACTIVE_FLOOR_DB = -60.0


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def resample(x: np.ndarray, orig_sr: int, target_sr: int = SR) -> np.ndarray:
    if orig_sr == target_sr:
        return x.astype(np.float32, copy=False)
    y = librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")
    return y.astype(np.float32, copy=False)


def load_16k(path: Path) -> np.ndarray:
    x, sr = load_mono(path)
    return resample(x, sr, SR)


def frame_rms_db(x: np.ndarray, frame_length: int = FRAME_LENGTH, hop_length: int = HOP_LENGTH) -> np.ndarray:
    rms = librosa.feature.rms(y=x, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def active_mask(
    x: np.ndarray,
    top_db: float = ACTIVE_TOP_DB,
    floor_db: float = ACTIVE_FLOOR_DB,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Frames whose RMS is within top_db of the loudest frame and above an absolute floor (dBFS)."""
    db = frame_rms_db(x, frame_length, hop_length)
    if db.size == 0:
        return np.zeros(0, dtype=bool)
    return db > max(float(db.max()) - top_db, floor_db)


def active_seconds(x: np.ndarray, sr: int = SR, hop_length: int = HOP_LENGTH) -> float:
    return float(active_mask(x, hop_length=hop_length).sum() * hop_length / sr)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
