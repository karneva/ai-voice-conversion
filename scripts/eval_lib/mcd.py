"""Mel-cepstral distortion for paired clips (held-out reconstruction only).

Built on mel-cepstral-distance 0.0.4 internals, with audio loaded by us:
- both signals at 16 kHz, peak-normalized (the package's norm_audio default)
- STFT: 32 ms Hann window, n_fft 32 ms (512 samples), hop 8 ms
- 20 mel bands over 0-8000 Hz, log10 energy (Bel), DCT to 20 mel-cepstra
- distance over coefficients 1..15 (s=1 drops c0 / loudness, D=16)
- frames aligned by index (RVC preserves timing), trimmed to the shorter clip
- only frames where the reference is within 40 dB of its loudest frame

MCD_k = ||c_ref[s:D] - c_out[s:D]||_2 per frame, averaged. This is the
package's scale, not the (10/ln10)*sqrt(2) "dB" scale: compare values only
within this project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from mel_cepstral_distance import get_mel_spectrogram, get_mfccs
from mel_cepstral_distance.computation import get_MCD_k, get_X_km


@dataclass(frozen=True)
class McdConfig:
    sample_rate: int = 16000
    n_fft_ms: float = 32.0
    win_ms: float = 32.0
    hop_ms: float = 8.0
    window: str = "hanning"
    n_mels: int = 20
    fmin: int = 0
    fmax: int = 8000
    s: int = 1
    D: int = 16
    top_db: float = 40.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _ms(ms: float, sr: int) -> int:
    return int(round(ms * sr / 1000.0))


def _spectra(x: np.ndarray, cfg: McdConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (mel-cepstra [coeffs, frames], frame energy in dB)."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    x = x.astype(np.float64) / peak if peak > 0 else x.astype(np.float64)
    sr = cfg.sample_rate
    amp = get_X_km(x, _ms(cfg.n_fft_ms, sr), _ms(cfg.win_ms, sr), _ms(cfg.hop_ms, sr), cfg.window)
    mel = get_mel_spectrogram(amp, sr, cfg.n_fft_ms, M=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    mfcc = get_mfccs(mel)
    energy_db = 10.0 * np.log10(np.sum(np.abs(amp) ** 2, axis=1) + 1e-12)
    return mfcc, energy_db


def mel_cepstral_distance(reference: np.ndarray, output: np.ndarray, config: McdConfig | None = None) -> float:
    cfg = config or McdConfig()
    ref_c, ref_db = _spectra(reference, cfg)
    out_c, _ = _spectra(output, cfg)
    n = min(ref_c.shape[1], out_c.shape[1])
    if n == 0:
        return float("nan")
    mask = ref_db[:n] > float(ref_db[:n].max()) - cfg.top_db
    mcd_k = get_MCD_k(ref_c[:, :n][:, mask], out_c[:, :n][:, mask], cfg.s, cfg.D)
    return float(np.mean(mcd_k)) if mcd_k.size else float("nan")
