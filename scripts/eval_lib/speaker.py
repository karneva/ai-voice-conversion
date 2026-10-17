"""ECAPA-TDNN speaker embeddings (SpeechBrain spkrec-ecapa-voxceleb)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from . import gpu

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000


class SpeakerEncoder:
    def __init__(self, models_dir: Path, device: str) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.device = "cuda:0" if device == "cuda" else device  # speechbrain wants an explicit index
        # COPY instead of symlinks: symlinks need admin/developer mode on Windows.
        self.model = EncoderClassifier.from_hparams(
            source=MODEL_ID,
            savedir=str(models_dir / "spkrec-ecapa-voxceleb"),
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )
        self.model.eval()

    def embed(self, wav16k: np.ndarray) -> np.ndarray:
        """L2-normalized 192-d embedding of one 16 kHz mono clip (tensor input, not a path)."""
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(wav16k, dtype=np.float32)).unsqueeze(0).to(self.device)
            emb = self.model.encode_batch(x).reshape(-1).double().cpu().numpy()
        return normalize(emb)

    def embed_batch(self, wavs16k: np.ndarray) -> np.ndarray:
        """L2-normalized embeddings (N, 192) of equal-length 16 kHz mono clips shaped (N, samples)."""
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(wavs16k, dtype=np.float32)).to(self.device)
            emb = self.model.encode_batch(x).reshape(x.shape[0], -1).double().cpu().numpy()
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    def close(self) -> None:
        del self.model
        gpu.release()


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def centroid(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    stack = np.stack([normalize(e) for e in embeddings])
    return normalize(stack.mean(axis=0))


def leave_one_out_cosines(embeddings: dict[str, np.ndarray]) -> dict[str, float]:
    """Cosine of each clip vs the centroid of all other clips."""
    if len(embeddings) < 2:
        raise ValueError("leave-one-out needs at least 2 clips")
    return {
        key: cosine(emb, centroid(v for k, v in embeddings.items() if k != key))
        for key, emb in embeddings.items()
    }
