"""F0 extraction with torchcrepe and F0 error metrics.

The converter uses RMVPE internally, so evaluation uses a different tracker
(CREPE 'full') to avoid grading the extractor with itself.

Voicing decision: periodicity (network confidence at the decoded bin) is
median-filtered over 3 frames; a frame is voiced when it is >= 0.21 (the
threshold from the torchcrepe README) AND the frame is active, i.e. its RMS is
within 40 dB of the clip's loudest frame and above -60 dBFS (audio.active_mask;
gain-invariant). torchcrepe's A-weighted Silence gate was not used because it
averages dB over all bins and gates out clean tonal frames. The same rule is
applied to reference and output so spurious voicing cancels out.

Decoding: Viterbi over the whole clip (torchcrepe's transition matrix) followed
by a +-4 bin weighted average around the path, without torchcrepe's random
dither, so identical audio gives identical F0. The network runs under fp16
autocast on CUDA (3.5x faster; measured <0.2 cents difference vs fp32).
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import gpu
from .audio import active_mask

SAMPLE_RATE = 16000
CENTS_OFFSET = 1997.3794084376191  # cents of CREPE bin 0 (relative to 10 Hz)


@functools.cache
def _transition(n_bins: int) -> np.ndarray:
    """torchcrepe's Viterbi transition: triangular, at most 11 bins (220 cents) per 10 ms frame."""
    xx, yy = np.meshgrid(range(n_bins), range(n_bins))
    transition = np.maximum(12 - np.abs(xx - yy), 0).astype(np.float64)
    return transition / transition.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class PitchConfig:
    model: str = "full"
    hop_length: int = 160  # 10 ms at 16 kHz
    fmin: float = 50.0
    fmax: float = 1100.0
    periodicity_threshold: float = 0.21
    median_window: int = 3
    batch_size: int = 512
    gross_error_cents: float = 50.0
    octave_error_cents: float = 600.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class F0Track:
    f0: np.ndarray  # Hz, NaN where unvoiced
    voiced: np.ndarray  # bool


class PitchExtractor:
    def __init__(self, device: str, config: PitchConfig | None = None) -> None:
        import torchcrepe

        self.torchcrepe = torchcrepe
        self.device = device
        self.config = config or PitchConfig()

    def extract(self, wav16k: np.ndarray) -> F0Track:
        tc, cfg = self.torchcrepe, self.config
        audio = torch.from_numpy(np.ascontiguousarray(wav16k, dtype=np.float32)).unsqueeze(0)
        probs = self._probabilities(audio)
        bins, f0 = self._decode(probs)
        periodicity = torch.from_numpy(probs[bins, np.arange(probs.shape[1])]).unsqueeze(0)
        periodicity = tc.filter.median(periodicity, cfg.median_window).squeeze(0).numpy()
        active = active_mask(wav16k, hop_length=cfg.hop_length)
        n = min(len(active), len(periodicity))
        active = np.pad(active[:n], (0, len(periodicity) - n))
        voiced = (periodicity >= cfg.periodicity_threshold) & active
        return F0Track(f0=np.where(voiced, f0, np.nan), voiced=voiced)

    def _probabilities(self, audio: torch.Tensor) -> np.ndarray:
        """Network output as [360 bins, frames]; frames are batched to bound VRAM."""
        tc, cfg = self.torchcrepe, self.config
        chunks = []
        autocast = torch.autocast("cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda"))
        with torch.inference_mode(), autocast:
            for frames in tc.preprocess(audio, SAMPLE_RATE, cfg.hop_length, cfg.batch_size, self.device, pad=True):
                chunks.append(tc.infer(frames, cfg.model, self.device).float().cpu())
        return torch.cat(chunks).numpy().T.astype(np.float64)

    def _decode(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Viterbi over the whole clip, then a +-4 bin weighted average around the path.

        torchcrepe's own bin->cents conversion adds random dither (+-20 cents),
        which makes identical audio score ~11 cents apart, so it is not used.
        """
        import librosa

        tc, cfg = self.torchcrepe, self.config
        n_bins = probs.shape[0]
        lo = int(tc.convert.frequency_to_bins(torch.tensor(cfg.fmin)))
        hi = int(tc.convert.frequency_to_bins(torch.tensor(cfg.fmax), torch.ceil))
        in_range = np.zeros(n_bins, dtype=bool)
        in_range[lo:hi] = True
        logits = np.where(in_range[:, None], probs, -np.inf)
        obs = np.exp(logits - logits.max(axis=0, keepdims=True))
        obs /= obs.sum(axis=0, keepdims=True)
        bins = librosa.sequence.viterbi(obs, _transition(n_bins)).astype(np.int64)

        cents_axis = tc.CENTS_PER_BIN * np.arange(n_bins) + CENTS_OFFSET
        raw = bins[None, :] + np.arange(-4, 5)[:, None]
        window = np.clip(raw, 0, n_bins - 1)
        valid = (raw == window) & in_range[window]
        weights = np.where(valid, probs[window, np.arange(probs.shape[1])[None, :]], 0.0)
        total = weights.sum(axis=0)
        cents = (weights * cents_axis[window]).sum(axis=0) / np.maximum(total, 1e-12)
        cents = np.where(total > 0, cents, cents_axis[bins])
        return bins, 10.0 * 2.0 ** (cents / 1200.0)

    def close(self) -> None:
        infer = self.torchcrepe.infer
        for attr in ("model", "capacity"):
            if hasattr(infer, attr):
                delattr(infer, attr)
        gpu.release()


def pitch_metrics(
    reference: F0Track,
    output: F0Track,
    transpose: int = 0,
    config: PitchConfig | None = None,
) -> dict[str, float | int]:
    """Compare output F0 with reference F0 * 2**(transpose/12); lengths are trimmed to the shorter track."""
    cfg = config or PitchConfig()
    n = min(len(reference.f0), len(output.f0))
    ref_v, out_v = reference.voiced[:n], output.voiced[:n]
    both = ref_v & out_v
    result: dict[str, float | int] = {
        "vuv_err": float(np.mean(ref_v != out_v)) if n else float("nan"),
        "voiced_frames_ref": int(ref_v.sum()),
        "voiced_frames_out": int(out_v.sum()),
        "voiced_frames_both": int(both.sum()),
        "f0_rmse_cents": float("nan"),
        "gpe50": float("nan"),
        "octave_err": float("nan"),
        "fpc": float("nan"),
    }
    if both.sum() < 2:
        return result
    ref_f0 = reference.f0[:n][both] * 2.0 ** (transpose / 12.0)
    out_f0 = output.f0[:n][both]
    cents = 1200.0 * np.log2(out_f0 / ref_f0)
    abs_cents = np.abs(cents)
    result["f0_rmse_cents"] = float(np.sqrt(np.mean(cents**2)))
    result["gpe50"] = float(np.mean(abs_cents > cfg.gross_error_cents))
    result["octave_err"] = float(np.mean(abs_cents > cfg.octave_error_cents))
    log_ref, log_out = np.log2(ref_f0), np.log2(out_f0)
    if np.std(log_ref) > 0 and np.std(log_out) > 0:
        result["fpc"] = float(np.corrcoef(log_ref, log_out)[0, 1])
    return result
