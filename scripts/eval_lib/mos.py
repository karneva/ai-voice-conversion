"""Reference-free naturalness predictors (optional, ``--mos``).

Both are trained on other data (UTMOS22: English speech; SingMOS-Pro: Chinese
and Japanese singing), so only deltas vs the unconverted source and
system-level averages are meaningful. Never report the numbers as "MOS".
"""

from __future__ import annotations

import sys
import types

import numpy as np
import torch

from . import gpu

SAMPLE_RATE = 16000
PREDICTORS: dict[str, tuple[str, str]] = {
    "utmos": ("tarepan/SpeechMOS:v1.2.0", "utmos22_strong"),
    "singmos": ("South-Twilight/SingMOS:v1.1.2", "singmos_pro"),
}


def _install_torchaudio_shims() -> None:
    """s3prl (used by SingMOS) calls APIs removed in torchaudio 2.9; inference never uses them."""
    import torchaudio

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None
    if "torchaudio.sox_effects" not in sys.modules:
        stub = types.ModuleType("torchaudio.sox_effects")

        def _unsupported(*args: object, **kwargs: object) -> None:
            raise NotImplementedError("torchaudio.sox_effects is not available in torchaudio>=2.9")

        stub.apply_effects_tensor = _unsupported
        stub.apply_effects_file = _unsupported
        sys.modules["torchaudio.sox_effects"] = stub
        torchaudio.sox_effects = stub


class MosPredictor:
    def __init__(self, name: str, device: str) -> None:
        if name not in PREDICTORS:
            raise ValueError(f"unknown predictor: {name}")
        repo, entry = PREDICTORS[name]
        if name == "singmos":
            _install_torchaudio_shims()
        self.name = name
        self.device = device
        self.model = torch.hub.load(repo, entry, trust_repo=True).to(device).eval()

    def score(self, wav16k: np.ndarray) -> float:
        wave = torch.from_numpy(np.ascontiguousarray(wav16k, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self.name == "singmos":
                lengths = torch.tensor([wave.shape[1]], device=self.device)
                value = self.model(wave, lengths)
            else:
                value = self.model(wave, SAMPLE_RATE)
        return float(value.reshape(-1)[0])

    def close(self) -> None:
        del self.model
        gpu.release()
