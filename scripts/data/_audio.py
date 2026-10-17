"""Shared path, CSV, audio IO, and frame-analysis helpers for the dataset tools."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
APPLIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")

MANIFEST_COLUMNS = ["file_id", "path", "song_id", "singer", "duration_s", "sr", "channels", "sha256", "split"]
CLIP_COLUMNS = ["clip_id", "path", "singer", "song_id", "start_s", "end_s", "source_path", "transpose", "lyrics"]

HOP_S = 0.01
ANALYSIS_SR = 16000
F0_FLOOR_HZ = 65.0
F0_CEIL_HZ = 1100.0
VOICING_GATE_DB = -50.0
ACTIVE_DB = -40.0

_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


# ---------------------------------------------------------------- paths / ids

def resolve_path(path: str | Path) -> Path:
    """Resolve a CLI or CSV path; relative paths are taken from the repo root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def record_path(path: Path) -> str:
    """Path string for CSV/JSON: repo-relative with '/', absolute if outside the repo."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def slugify(text: str, max_len: int = 60) -> str:
    """Filename-safe id. Keeps Unicode word characters (Korean included)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    text = text[:max_len].rstrip("._-") or "x"
    return f"{text}_" if text.upper() in _WINDOWS_RESERVED else text


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_hidden(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def iter_audio_files(
    input_dir: Path,
    pattern: str = "**/*",
    excludes: list[str] | None = None,
) -> list[Path]:
    """Audio files under input_dir matching pattern, skipping hidden dirs and exclude globs."""
    excludes = excludes or []
    files = []
    for path in input_dir.glob(pattern):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        relative = path.relative_to(input_dir)
        if is_hidden(relative):
            continue
        posix = relative.as_posix()
        if any(fnmatch.fnmatch(posix, ex) for ex in excludes):
            continue
        files.append(path)
    return sorted(files)


def song_name(relative: Path, song_level: int | None) -> str:
    if song_level is None:
        return relative.stem
    directories = relative.parts[:-1]
    if song_level < 0 or song_level >= len(directories):
        raise SystemExit(f"--song-level {song_level} is not a folder level of: {relative.as_posix()}")
    return directories[song_level]


# ---------------------------------------------------------------------- CSV

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write atomically so an interrupted run never leaves a truncated CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


# ------------------------------------------------------------------ audio IO

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return float32 samples shaped (frames, channels) and the sample rate."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, int(sr)
    except (sf.SoundFileError, RuntimeError):
        return _load_with_av(path)


def _load_with_av(path: Path) -> tuple[np.ndarray, int]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        sr = int(stream.codec_context.sample_rate)
        channels = int(stream.codec_context.channels)
        resampler = av.AudioResampler(format="flt", layout=stream.codec_context.layout.name, rate=sr)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1, channels))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1, channels))
    if not chunks:
        raise RuntimeError(f"No audio decoded from {path}")
    return np.concatenate(chunks).astype(np.float32), sr


def audio_info(path: Path) -> tuple[float, int, int]:
    """Duration in seconds, sample rate, channel count."""
    try:
        info = sf.info(str(path))
        if info.frames > 0:
            return info.frames / info.samplerate, int(info.samplerate), int(info.channels)
    except (sf.SoundFileError, RuntimeError):
        pass
    data, sr = _load_with_av(path)
    return data.shape[0] / sr, sr, data.shape[1]


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim == 2 else data


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr, subtype="PCM_24")


def db(value: np.ndarray | float, floor: float = 1e-10) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(value, floor))


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    import librosa

    return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq", axis=0)


# ------------------------------------------------------------ frame analysis

@dataclass
class FrameAnalysis:
    """Per-frame RMS level and F0 on a HOP_S grid (frame i is centred at i * HOP_S)."""

    rms_db: np.ndarray
    f0: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.f0)

    @property
    def voiced(self) -> np.ndarray:
        return self.f0 > 0

    @property
    def active(self) -> np.ndarray:
        return self.rms_db > ACTIVE_DB

    def frame(self, seconds: float) -> int:
        return int(round(seconds / HOP_S))

    def voiced_fraction(self, start: int, end: int) -> float:
        return float(self.voiced[start:end].mean()) if end > start else 0.0


def frame_rms_db(mono: np.ndarray, sr: int, window_s: float = 0.04) -> np.ndarray:
    hop = int(round(sr * HOP_S))
    win = int(round(sr * window_s))
    padded = np.pad(mono.astype(np.float64), (win // 2, win // 2))
    n_frames = 1 + len(mono) // hop
    squares = np.concatenate([[0.0], np.cumsum(padded**2)])
    starts = np.arange(n_frames) * hop
    ends = np.minimum(starts + win, len(padded))
    energy = (squares[ends] - squares[starts]) / np.maximum(ends - starts, 1)
    return db(np.sqrt(energy)).astype(np.float32)


def estimate_f0(mono: np.ndarray, sr: int, method: str = "harvest") -> np.ndarray:
    """pyworld F0 in Hz on the HOP_S grid; 0 marks unvoiced frames."""
    import pyworld

    x = resample(mono.astype(np.float64), sr, ANALYSIS_SR).astype(np.float64)
    period_ms = HOP_S * 1000.0
    if method == "harvest":
        f0, _ = pyworld.harvest(x, ANALYSIS_SR, f0_floor=F0_FLOOR_HZ, f0_ceil=F0_CEIL_HZ, frame_period=period_ms)
    elif method == "dio":
        f0, t = pyworld.dio(x, ANALYSIS_SR, f0_floor=F0_FLOOR_HZ, f0_ceil=F0_CEIL_HZ, frame_period=period_ms)
        f0 = pyworld.stonemask(x, f0, t, ANALYSIS_SR)
    else:
        raise ValueError(f"Unknown F0 method: {method}")
    return f0.astype(np.float32)


def analyze_frames(mono: np.ndarray, sr: int, f0_method: str = "harvest") -> FrameAnalysis:
    rms = frame_rms_db(mono, sr)
    f0 = estimate_f0(mono, sr, f0_method)
    n = min(len(rms), len(f0))
    rms, f0 = rms[:n], f0[:n].copy()
    f0[rms < VOICING_GATE_DB] = 0.0
    return FrameAnalysis(rms_db=rms, f0=f0)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges where mask is True."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def trim_inactive(analysis: FrameAnalysis, start: int, end: int, pad_frames: int = 10) -> tuple[int, int]:
    """Shrink [start, end) to the active frames inside it, keeping a small pad."""
    active = np.flatnonzero(analysis.active[start:end])
    if active.size == 0:
        return start, start
    new_start = max(start, start + int(active[0]) - pad_frames)
    new_end = min(end, start + int(active[-1]) + 1 + pad_frames)
    return new_start, new_end


LYRIC_PLACEHOLDERS = {"", "<SP>", "<AP>"}


def annotated_lyrics(source: Path, start_s: float, end_s: float) -> str:
    """Lyrics of [start_s, end_s) from a GTSinger-style <stem>.json word list next to the audio, else ''.

    A word belongs to the clip when its midpoint falls inside it; <SP>/<AP> markers are dropped.
    """
    annotation = source.with_suffix(".json")
    if not annotation.exists():
        return ""
    try:
        words = json.loads(annotation.read_text(encoding="utf-8"))
        return "".join(
            str(w["word"]).strip()
            for w in words
            if str(w.get("word", "")).strip() not in LYRIC_PLACEHOLDERS
            and start_s <= (float(w["start_time"]) + float(w["end_time"])) / 2 < end_s
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"warning: cannot read lyrics from {record_path(annotation)}: {error}")
        return ""


def median_f0(f0: np.ndarray) -> float:
    voiced = f0[f0 > 0]
    return float(np.median(voiced)) if voiced.size else float("nan")


def setup_console() -> None:
    """Print UTF-8 (Korean file names) without crashing on legacy Windows code pages."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass
