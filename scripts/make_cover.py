#!/usr/bin/env python3
"""Make a listenable cover from one song file with a trained RVC model (Applio 3.6.6).

Steps: (1) vocal separation (reuses dataset/*/separated/**/ stems of the same song, else runs
scripts/separate_vocals.py into dataset/<track>/separated/_cache/), (2) optional transpose
estimate from a reference performance (median-F0 ratio for the octave, mean-chroma circular
cross-correlation for the pitch class), (3) Applio `core.py infer` on the separated vocals,
(4) instrumental pitch shift by the transpose wrapped to [-6, 5] semitones (ffmpeg rubberband,
tempo preserved), (5) mix with scripts/data/mix_cover.py helpers: converted vocals are level-matched
to the original vocal stem, then the mix is normalized to -14 LUFS and peak-limited to -1 dBFS.

Writes <output> and <output>.json (settings, input hashes, estimates, timings, measurements).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR / "data"))

import mix_cover  # noqa: E402
from _audio import (  # noqa: E402
    REPO_ROOT,
    audio_info,
    db,
    load_audio,
    record_path,
    resample,
    resolve_path,
    setup_console,
    sha256_file,
    slugify,
    to_mono,
    write_wav,
)
from run_experiment import (  # noqa: E402
    APPLIO_DIR,
    APPLIO_LOGS,
    APPLIO_PY,
    FAILURE_PATTERNS,
    INDEX_PATH_TRAP,
    LineScanner,
    format_command,
    model_embedder,
    run_logged,
)

VERSION = "make_cover/1"
VOCALS_MARK = "_(vocals)_"
OTHER_MARK = "_(other)_"
SEPARATOR_PY = REPO_ROOT / "separator" / "env" / "python.exe"
SEPARATE_SCRIPT = SCRIPTS_DIR / "separate_vocals.py"
DURATION_TOLERANCE_S = 0.5
EMBEDDERS = ("contentvec", "spin", "spin-v2", "chinese-hubert-base", "japanese-hubert-base", "korean-hubert-base")
F0_METHODS = ("rmvpe", "crepe", "fcpe")  # Applio's hybrid[...] choices crash in 3.6.6
CHROMA_SR = 16000
CHROMA_HOP = 512
LAG_HOP_S = 0.005
LAG_MAX_S = 0.5
LAG_WARN_MS = 20.0
F0_CHROMA_WARN_ST = 3.0
LOSSY_MARGIN_DB = 0.3  # MP3/Vorbis decoding overshoots the limiter by ~0.1 dB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Song file -> finished cover: separate, (estimate transpose), convert with Applio, shift, mix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--song", required=True, help="Song to cover (any format ffmpeg/libav decodes).")
    model = parser.add_argument_group("model (give --pth, or --run with --epoch)")
    model.add_argument("--pth", help="Applio weight file (.pth).")
    model.add_argument("--run", help="Run id: uses Applio/logs/<run>/<run>_<epoch>e_*s.pth.")
    model.add_argument("--epoch", type=int, help="Epoch of the --run checkpoint.")
    model.add_argument("--index", help="FAISS index. Default: <model>.index next to the weights.")
    model.add_argument(
        "--embedder",
        choices=EMBEDDERS,
        help="Embedder used in training. Default: model_info.json next to the weights, else contentvec.",
    )
    pitch = parser.add_mutually_exclusive_group()
    pitch.add_argument("--transpose", type=int, help="Semitones applied to the vocals (-24..24). Default 0.")
    pitch.add_argument(
        "--reference",
        help="The target singer's own performance of this song (song file or its *_(vocals)_*.wav); "
        "the transpose is estimated from it.",
    )
    parser.add_argument(
        "--pitch-tracker",
        choices=("crepe", "harvest"),
        default="crepe",
        help="F0 tracker for the transpose estimate (crepe uses the GPU when available).",
    )
    parser.add_argument("--estimate-only", action="store_true", help="Stop after the transpose estimate.")
    infer = parser.add_argument_group("conversion (Applio infer)")
    infer.add_argument("--f0-method", choices=F0_METHODS, default="rmvpe")
    infer.add_argument(
        "--split-audio",
        action="store_true",
        help="Applio --split-audio: convert silence-separated chunks (less VRAM, hard cuts at -60 dB silences).",
    )
    infer.add_argument(
        "--index-rate", type=float, default=0.75, help="Index influence; higher = closer timbre, blurrier diction."
    )
    infer.add_argument(
        "--protect", type=float, default=0.33, help="Consonant/breath protection, 0..0.5 (0.5 turns it off)."
    )
    infer.add_argument(
        "--volume-envelope",
        type=float,
        default=1.0,
        help="1 keeps the converted loudness contour; lower values pull it toward the source vocals.",
    )
    mix = parser.add_argument_group("mix")
    mix.add_argument(
        "--instrumental-shift",
        default="auto",
        help="Semitones for the instrumental, or 'auto' = transpose wrapped to [-6, 5] (octaves need no shift).",
    )
    mix.add_argument(
        "--vocal-gain-db", type=float, default=0.0, help="Vocal gain after matching the original vocal level."
    )
    mix.add_argument("--match-lufs", type=float, default=-14.0, help="Integrated loudness of the final mix.")
    mix.add_argument("--ceiling-db", type=float, default=-1.0, help="Sample-peak ceiling of the final mix (dBFS).")
    parser.add_argument(
        "--output",
        help="Output .mp3/.wav/.flac/.ogg. Default: dataset/private/covers/<song_id>__<run>_<E>e_t<N>.mp3",
    )
    parser.add_argument(
        "--keep-intermediates", action="store_true", help="Keep <output stem>_work/ (Applio input/output, logs)."
    )
    args = parser.parse_args()
    if bool(args.pth) == bool(args.run):
        parser.error("give exactly one of --pth or --run")
    if args.run and args.epoch is None:
        parser.error("--run needs --epoch")
    if args.transpose is not None and not -24 <= args.transpose <= 24:
        parser.error("--transpose must be within -24..24 (Applio's range)")
    if not 0 <= args.index_rate <= 1 or not 0 <= args.protect <= 0.5 or not 0 <= args.volume_envelope <= 1:
        parser.error("--index-rate and --volume-envelope must be 0..1, --protect 0..0.5")
    if args.instrumental_shift != "auto" and not re.fullmatch(r"[+-]?\d+", args.instrumental_shift):
        parser.error("--instrumental-shift must be 'auto' or an integer")
    return args


# ------------------------------------------------------------------ model


@dataclass
class Model:
    pth: Path
    index: Path | None
    tag: str
    embedder: str
    embedder_custom: str | None
    embedder_source: str


def resolve_model(args: argparse.Namespace) -> Model:
    if args.run:
        folder = APPLIO_LOGS / args.run
        found = sorted(folder.glob(f"{args.run}_{args.epoch}e_*s.pth"))
        if len(found) != 1:
            raise SystemExit(f"Expected one {args.run}_{args.epoch}e_*s.pth in {record_path(folder)}, found {len(found)}")
        pth, name, tag = found[0], args.run, f"{args.run}_{args.epoch}e"
    else:
        pth = resolve_path(args.pth)
        if not pth.is_file():
            raise SystemExit(f"Weights not found: {args.pth}")
        match = re.fullmatch(r"(.+)_(\d+)e_\d+s", pth.stem)
        name = match.group(1) if match else pth.stem
        tag = f"{name}_{match.group(2)}e" if match else slugify(pth.stem)
    index = resolve_path(args.index) if args.index else find_index(pth.parent, name)
    if index is not None and not index.is_file():
        raise SystemExit(f"Index not found: {args.index}")
    embedder, custom, source = resolve_embedder(pth.parent, args.embedder)
    return Model(pth, index, tag, embedder, custom, source)


def find_index(folder: Path, name: str) -> Path | None:
    preferred = folder / f"{name}.index"
    if preferred.is_file():
        return preferred
    found = sorted(folder.glob("*.index"))
    if len(found) == 1:
        return found[0]
    print(f"warning: no single index in {record_path(folder)} ({len(found)} found); converting without index")
    return None


def resolve_embedder(model_dir: Path, requested: str | None) -> tuple[str, str | None, str]:
    """(embedder, custom path, where it came from). Refuses an embedder that differs from training."""
    trained = model_embedder(model_dir)
    if trained is None:
        return requested or "contentvec", None, "argument" if requested else "default (no model_info.json)"
    if trained not in EMBEDDERS:
        if requested:
            raise SystemExit(f"Model was trained with custom embedder {trained!r}; drop --embedder.")
        return "custom", trained, "model_info.json"
    if requested and requested != trained:
        raise SystemExit(f"--embedder {requested} differs from the training embedder {trained!r} (model_info.json)")
    return trained, None, "model_info.json"


# ------------------------------------------------------------- separation


@dataclass
class Stems:
    vocals: Path
    instrumental: Path | None
    origin: str


def name_key(text: str) -> str:
    """Comparable song name: Windows/audio-separator rewrite '__', trailing dots, etc., so keep only letters/digits."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFC", text)).casefold()


def separated_roots() -> list[Path]:
    return sorted(p for p in (REPO_ROOT / "dataset").glob("*/separated") if p.is_dir())


def find_stems(song: Path, duration: float) -> Stems | None:
    key = name_key(song.stem)
    if not key:
        return None
    candidates = []
    for root in separated_roots():
        for vocals in sorted(root.rglob(f"*{VOCALS_MARK}*.wav")):
            if name_key(vocals.name.split(VOCALS_MARK)[0]) != key:
                continue
            other = vocals.with_name(vocals.name.replace(VOCALS_MARK, OTHER_MARK, 1))
            stem_duration = audio_info(vocals)[0]
            if not other.is_file() or abs(stem_duration - duration) > DURATION_TOLERANCE_S:
                print(f"separation: skip {record_path(vocals)} (no instrumental or {stem_duration:.1f}s != {duration:.1f}s)")
                continue
            candidates.append(Stems(vocals, other, "reused"))
    if len(candidates) > 1:
        print(f"separation: {len(candidates)} matching stem pairs, using the first:")
        for c in candidates:
            print(f"  {record_path(c.vocals)}")
    return candidates[0] if candidates else None


def cache_dir(song: Path) -> Path:
    try:
        parts = song.resolve().relative_to(REPO_ROOT / "dataset").parts
        track = parts[0] if len(parts) > 1 else "private"
    except ValueError:
        track = "private"
    return REPO_ROOT / "dataset" / track / "separated" / "_cache"


def find_ffmpeg() -> Path | None:
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet"
    candidates = [winget / "Links" / "ffmpeg.exe", *sorted(winget.glob("Packages/*FFmpeg*/*/bin/ffmpeg.exe"))]
    return next((c for c in candidates if c.is_file()), None)


def env_with_ffmpeg(ffmpeg: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if ffmpeg is not None:
        env["PATH"] = f"{ffmpeg.parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def separate(song: Path, ffmpeg: Path | None) -> Stems:
    duration = audio_info(song)[0]
    stems = find_stems(song, duration)
    if stems:
        print(f"separation: reuse {record_path(stems.vocals)}")
        return stems
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found (PATH or WinGet); separate_vocals.py needs it")
    target = cache_dir(song)
    command = [str(SEPARATOR_PY), str(SEPARATE_SCRIPT), str(song), str(target)]
    print(f"separation: no stems for {song.name}; running\n  {format_command(command)}")
    subprocess.run(command, check=True, env=env_with_ffmpeg(ffmpeg))
    stems = find_stems(song, duration)
    if stems is None:
        raise SystemExit(f"Separation finished but no matching *{VOCALS_MARK}*.wav appeared in {record_path(target)}")
    stems.origin = "separated"
    return stems


def reference_vocals(path: Path, ffmpeg: Path | None) -> Stems:
    if VOCALS_MARK in path.name:
        return Stems(path, None, "given")
    return separate(path, ffmpeg)


# -------------------------------------------------------- transpose estimate


def load_16k(path: Path) -> np.ndarray:
    data, sr = load_audio(path)
    return resample(to_mono(data), sr, CHROMA_SR).astype(np.float32)


def voiced_f0(x16: np.ndarray, tracker: str, extractor: Any) -> np.ndarray:
    if tracker == "crepe":
        track = extractor.extract(x16)
        return track.f0[track.voiced].astype(np.float64)
    from _audio import analyze_frames

    f0 = analyze_frames(x16, CHROMA_SR, "harvest").f0
    return f0[f0 > 0].astype(np.float64)


def f0_summary(f0: np.ndarray) -> dict[str, float]:
    p10, p50, p90 = np.percentile(f0, [10, 50, 90])
    return {"voiced_s": round(len(f0) * 0.01, 1), "median_hz": round(p50, 2), "p10_hz": round(p10, 2), "p90_hz": round(p90, 2)}


def mean_chroma(x16: np.ndarray) -> np.ndarray:
    """Mean CQT chroma over active frames (librosa estimates and removes each track's tuning)."""
    import librosa

    from eval_lib.audio import active_mask

    chroma = librosa.feature.chroma_cqt(y=x16, sr=CHROMA_SR, hop_length=CHROMA_HOP)
    active = active_mask(x16, hop_length=CHROMA_HOP, frame_length=4 * CHROMA_HOP)
    n = min(chroma.shape[1], len(active))
    frames = chroma[:, :n][:, active[:n]]
    return (frames if frames.shape[1] else chroma).mean(axis=1)


def pitch_class_profile(f0: np.ndarray) -> np.ndarray:
    """12-bin histogram of voiced F0 pitch classes after removing the track's mean tuning offset."""
    cents = 1200.0 * np.log2(f0 / 440.0)
    angle = 2 * np.pi * (cents % 100.0) / 100.0
    tuning = 100.0 * np.angle(np.mean(np.exp(1j * angle))) / (2 * np.pi)
    classes = np.round((cents - tuning) / 100.0).astype(int) % 12
    return np.bincount(classes, minlength=12).astype(np.float64) / max(len(classes), 1)


def circular_offset(source: np.ndarray, reference: np.ndarray) -> tuple[int, list[float]]:
    """Shift k (semitones, 0..11) maximizing corr(roll(source, k), reference); also the 12 correlations."""
    scores = [float(np.corrcoef(np.roll(source, k), reference)[0, 1]) for k in range(12)]
    return int(np.argmax(scores)), scores


def wrap(semitones: float) -> int:
    """Nearest-to-zero representative mod 12, in [-6, 5]."""
    return int((round(semitones) + 6) % 12 - 6)


def estimate_transpose(source_vocals: Path, ref_vocals: Path, tracker: str) -> dict[str, Any]:
    print(f"transpose: analysing {record_path(source_vocals)} vs {record_path(ref_vocals)} ({tracker})")
    tracks = {"source": load_16k(source_vocals), "reference": load_16k(ref_vocals)}
    extractor = None
    if tracker == "crepe":
        from eval_lib.gpu import pick_device
        from eval_lib.pitch import PitchExtractor

        extractor = PitchExtractor(pick_device())
    try:
        f0 = {name: voiced_f0(x, tracker, extractor) for name, x in tracks.items()}
    finally:
        if extractor is not None:
            extractor.close()
    if min(len(v) for v in f0.values()) < 100:
        raise SystemExit("Too few voiced frames to estimate the transpose; pass --transpose instead")

    summary = {name: f0_summary(v) for name, v in f0.items()}
    f0_st = 12.0 * np.log2(summary["reference"]["median_hz"] / summary["source"]["median_hz"])
    chroma_k, chroma_scores = circular_offset(mean_chroma(tracks["source"]), mean_chroma(tracks["reference"]))
    pcp_k, pcp_scores = circular_offset(pitch_class_profile(f0["source"]), pitch_class_profile(f0["reference"]))
    chosen = chroma_k + 12 * int(round((f0_st - chroma_k) / 12.0))
    chosen = int(np.clip(chosen, -24, 24))

    ranked = sorted(range(12), key=lambda k: -chroma_scores[k])
    result = {
        "tracker": tracker,
        "source": summary["source"],
        "reference": summary["reference"],
        "f0_median_semitones": round(float(f0_st), 2),
        "chroma_offset": wrap(chroma_k),
        "chroma_scores": [round(s, 4) for s in chroma_scores],
        "chroma_margin": round(chroma_scores[ranked[0]] - chroma_scores[ranked[1]], 4),
        "f0_pitch_class_offset": wrap(pcp_k),
        "f0_pitch_class_scores": [round(s, 4) for s in pcp_scores],
        "transpose": chosen,
    }
    print_estimate(result, ranked)
    return result


def print_estimate(est: dict[str, Any], ranked: list[int]) -> None:
    src, ref = est["source"], est["reference"]
    print(f"  F0 median: source {src['median_hz']:.1f} Hz ({src['voiced_s']:.0f}s voiced), "
          f"reference {ref['median_hz']:.1f} Hz ({ref['voiced_s']:.0f}s voiced) -> {est['f0_median_semitones']:+.2f} st")
    top = ", ".join(f"{wrap(k):+d}:{est['chroma_scores'][k]:.3f}" for k in ranked[:3])
    print(f"  chroma key offset: {est['chroma_offset']:+d} st mod 12 (top corr {top}; margin {est['chroma_margin']:.3f})")
    print(f"  F0 pitch-class offset (cross-check): {est['f0_pitch_class_offset']:+d} st mod 12")
    gap = est["f0_median_semitones"] - est["transpose"]
    print(f"  -> transpose {est['transpose']:+d}: pitch class from chroma, octave from the F0 ratio "
          f"(F0 ratio is {gap:+.2f} st from it)")
    if abs(gap) > F0_CHROMA_WARN_ST:
        print(f"  warning: F0 ratio and chroma disagree by {gap:+.1f} st; the octave choice is uncertain")
    if est["f0_pitch_class_offset"] != est["chroma_offset"]:
        print("  warning: chroma and F0 pitch-class offsets differ; check by ear or pass --transpose")
    if round(est["f0_median_semitones"]) % 12 != est["chroma_offset"] % 12:
        print("  note: rounding the F0 ratio alone would give a different key than chroma (sections may differ)")


# --------------------------------------------------------------- conversion


def check_trap(paths: list[Path]) -> None:
    for path in paths:
        if INDEX_PATH_TRAP in str(path):
            raise SystemExit(f"Path contains '{INDEX_PATH_TRAP}' (Applio rewrites it to 'added'): {path}")


def infer_command(model: Model, source: Path, output: Path, transpose: int, args: argparse.Namespace) -> list[str]:
    command = [
        str(APPLIO_PY), "core.py", "infer",
        "--input-path", str(source),
        "--output-path", str(output),
        "--pth-path", str(model.pth),
        "--index-path", str(model.index) if model.index else "",
        "--f0-method", args.f0_method,
        "--pitch", str(transpose),
        "--index-rate", f"{args.index_rate if model.index else 0.0}",
        "--protect", f"{args.protect}",
        "--volume-envelope", f"{args.volume_envelope}",
        "--embedder-model", model.embedder,
        "--export-format", "WAV",
        "--sid", "0",
    ]
    if args.split_audio:
        command.append("--split-audio")
    if model.embedder_custom:
        command += ["--embedder-model-custom", model.embedder_custom]
    return command


def convert(model: Model, vocals: Path, work: Path, transpose: int, args: argparse.Namespace) -> tuple[Path, list[str]]:
    source, converted = work / "vocals_in.wav", work / "vocals_converted.wav"
    shutil.copyfile(vocals, source)  # ASCII path for Applio; the stem name may hold emoji/brackets
    converted.unlink(missing_ok=True)
    command = infer_command(model, source, converted, transpose, args)
    check_trap([model.pth, *([model.index] if model.index else []), source, converted])
    print(f"convert: {format_command(command)}")
    scanner = LineScanner(FAILURE_PATTERNS["infer"])
    returncode = run_logged(command, work / "applio_infer.log", scanner)
    if returncode != 0 or scanner.matches or not converted.is_file() or converted.stat().st_size == 0:
        details = "\n  ".join(scanner.matches[:10]) or "(no failure text matched)"
        raise SystemExit(f"Applio infer failed (exit {returncode}); see {record_path(work / 'applio_infer.log')}\n  {details}")
    return converted, command


# ---------------------------------------------------- instrumental shift


def has_rubberband(ffmpeg: Path) -> bool:
    result = subprocess.run([str(ffmpeg), "-hide_banner", "-filters"], capture_output=True, text=True, errors="replace")
    return any(line.split()[1:2] == ["rubberband"] for line in result.stdout.splitlines() if line.strip())


def shift_instrumental(source: Path, target: Path, semitones: int, ffmpeg: Path | None) -> str:
    ratio = 2.0 ** (semitones / 12.0)
    if ffmpeg is not None and has_rubberband(ffmpeg):
        filt = f"rubberband=pitch={ratio:.10f}:tempo=1:pitchq=quality:channels=together"
        command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                   "-af", filt, "-c:a", "pcm_f32le", str(target)]
        subprocess.run(command, check=True)
        return f"ffmpeg {filt}"
    import librosa

    data, sr = load_audio(source)
    shifted = np.stack([librosa.effects.pitch_shift(data[:, c], sr=sr, n_steps=semitones, res_type="soxr_hq")
                        for c in range(data.shape[1])], axis=1)
    write_wav(target, shifted, sr)
    return f"librosa.effects.pitch_shift n_steps={semitones}"


# --------------------------------------------------------------------- mix


def fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    return samples[:length] if samples.shape[0] >= length else mix_cover.pad_to(samples, length)


def onset_envelope(samples: np.ndarray, sr: int) -> np.ndarray:
    hop = int(round(sr * LAG_HOP_S))
    mono = to_mono(samples)
    n = len(mono) // hop
    level = db(np.sqrt(np.mean(mono[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1)))
    return np.maximum(np.diff(np.maximum(level, -80.0)), 0.0)


def lag_ms(reference: np.ndarray, other: np.ndarray, sr: int) -> float:
    """Delay of `other` relative to `reference` from onset-envelope cross-correlation (+ = later)."""
    a, b = onset_envelope(reference, sr), onset_envelope(other, sr)
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    max_lag = int(LAG_MAX_S / LAG_HOP_S)
    scores = [float(np.dot(a[max(0, -k): n - max(0, k)], b[max(0, k): n - max(0, -k)])) for k in range(-max_lag, max_lag + 1)]
    return (int(np.argmax(scores)) - max_lag) * LAG_HOP_S * 1000.0


def limiter_ceiling_db(args: argparse.Namespace) -> float:
    lossy = Path(args.output or ".mp3").suffix.lower() in (".mp3", ".ogg")
    return args.ceiling_db - (LOSSY_MARGIN_DB if lossy else 0.0)


def mix_cover_audio(
    converted: Path, original_vocals: Path, instrumental: Path, length: int, args: argparse.Namespace
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Mix trimmed/padded to `length` frames (the unshifted instrumental, i.e. the song length)."""
    inst, sr = load_audio(instrumental)
    inst_len = inst.shape[0] / sr
    inst = fit_length(inst, length)
    original, orig_sr = load_audio(original_vocals)
    original = resample(original, orig_sr, sr)
    voc, voc_sr = load_audio(converted)
    raw_len = voc.shape[0] / voc_sr
    voc = fit_length(mix_cover.match_channels(resample(voc, voc_sr, sr), inst.shape[1]), inst.shape[0])

    level_db = mix_cover.loudness(original, sr) - mix_cover.loudness(voc, sr)
    level_db = level_db if np.isfinite(level_db) else 0.0
    mix = voc.astype(np.float64) * 10 ** ((level_db + args.vocal_gain_db) / 20) + inst.astype(np.float64)
    before = mix_cover.loudness(mix, sr)
    if np.isfinite(before):
        mix *= 10 ** ((args.match_lufs - before) / 20)
    mix = mix_cover.limit(mix, sr, 10 ** (limiter_ceiling_db(args) / 20))
    stats = {
        "converted_vocals_s": round(raw_len, 3),
        "instrumental_s": round(inst_len, 3),
        "vocal_level_match_db": round(float(level_db), 2),
        "vocal_lag_ms": lag_ms(original, voc, sr),
        "mix_loudness_before_lufs": round(before, 2),
    }
    return mix.astype(np.float32), sr, stats


def measure(path: Path) -> dict[str, float]:
    data, sr = load_audio(path)
    return {
        "duration_s": round(data.shape[0] / sr, 3),
        "sr": sr,
        "loudness_lufs": round(mix_cover.loudness(data.astype(np.float64), sr), 2),
        "peak_dbfs": round(float(db(np.abs(data).max())), 2),
    }


# ------------------------------------------------------------------- main


def default_output(song: Path, model: Model, transpose: int) -> Path:
    return REPO_ROOT / "dataset" / "private" / "covers" / f"{slugify(song.stem)}__{model.tag}_t{transpose:+d}.mp3"


def instrumental_semitones(spec: str, transpose: int) -> int:
    return wrap(transpose) if spec == "auto" else int(spec)


def hashes(paths: dict[str, Path | None]) -> dict[str, Any]:
    return {k: {"path": record_path(p), "sha256": sha256_file(p)} for k, p in paths.items() if p is not None}


def main() -> int:
    setup_console()
    args = parse_args()
    timings: dict[str, float] = {}
    started = time.perf_counter()
    song = resolve_path(args.song)
    if not song.is_file():
        raise SystemExit(f"Song not found: {args.song}")
    model = resolve_model(args)
    print(f"model: {record_path(model.pth)} | index {record_path(model.index) if model.index else '-'} | "
          f"embedder {model.embedder} ({model.embedder_source})")
    ffmpeg = find_ffmpeg()

    t = time.perf_counter()
    stems = separate(song, ffmpeg)
    reference = reference_vocals(resolve_path(args.reference), ffmpeg) if args.reference else None
    timings["separation_s"] = round(time.perf_counter() - t, 1)

    estimate = None
    if reference is not None:
        t = time.perf_counter()
        estimate = estimate_transpose(stems.vocals, reference.vocals, args.pitch_tracker)
        timings["transpose_estimate_s"] = round(time.perf_counter() - t, 1)
        transpose = estimate["transpose"]
    else:
        transpose = args.transpose or 0
        print(f"transpose: {transpose:+d} (given)")
    inst_shift = instrumental_semitones(args.instrumental_shift, transpose)
    if (transpose - inst_shift) % 12:
        print(f"warning: vocals {transpose:+d} st vs instrumental {inst_shift:+d} st are not an octave apart; they will clash")
    if args.estimate_only:
        return 0

    output = resolve_path(args.output) if args.output else default_output(song, model, transpose)
    if output.suffix.lower() not in mix_cover.OUTPUT_FORMATS:
        raise SystemExit(f"Unsupported output extension {output.suffix} (use {', '.join(mix_cover.OUTPUT_FORMATS)})")
    work = output.with_name(f"{output.stem}_work")
    work.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    converted, command = convert(model, stems.vocals, work, transpose, args)
    timings["conversion_s"] = round(time.perf_counter() - t, 1)

    instrumental, shift_method = stems.instrumental, "none"
    if inst_shift:
        t = time.perf_counter()
        instrumental = work / f"instrumental_{inst_shift:+d}.wav"
        shift_method = shift_instrumental(stems.instrumental, instrumental, inst_shift, ffmpeg)
        timings["instrumental_shift_s"] = round(time.perf_counter() - t, 1)
        print(f"instrumental: shifted {inst_shift:+d} st with {shift_method}")
    else:
        print("instrumental: no shift (transpose is a whole number of octaves)")

    t = time.perf_counter()
    orig_inst, inst_sr = load_audio(stems.instrumental)
    mix, sr, mix_stats = mix_cover_audio(converted, stems.vocals, instrumental, orig_inst.shape[0], args)
    if inst_shift:
        mix_stats["instrumental_lag_ms"] = lag_ms(orig_inst, fit_length(load_audio(instrumental)[0], len(orig_inst)), inst_sr)
    mix_cover.write_output(output, mix, sr)
    timings["mix_s"] = round(time.perf_counter() - t, 1)
    timings["total_s"] = round(time.perf_counter() - started, 1)

    song_duration = audio_info(song)[0]
    result = measure(output)
    result["song_duration_s"] = round(song_duration, 3)
    result["duration_diff_s"] = round(result["duration_s"] - song_duration, 3)
    for key in ("vocal_lag_ms", "instrumental_lag_ms"):
        if abs(mix_stats.get(key, 0.0)) > LAG_WARN_MS:
            print(f"warning: {key} = {mix_stats[key]:+.0f} ms; vocals and backing may be out of sync")

    sidecar = output.with_suffix(".json")
    record = {
        "version": VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output": record_path(output),
        "settings": {
            "model_tag": model.tag,
            "embedder": model.embedder_custom or model.embedder,
            "embedder_source": model.embedder_source,
            "f0_method": args.f0_method,
            "index_rate": args.index_rate if model.index else 0.0,
            "protect": args.protect,
            "volume_envelope": args.volume_envelope,
            "transpose": transpose,
            "transpose_source": "reference" if estimate else "argument",
            "instrumental_shift": inst_shift,
            "instrumental_shift_method": shift_method,
            "vocal_gain_db": args.vocal_gain_db,
            "match_lufs": args.match_lufs,
            "ceiling_db": args.ceiling_db,
            "limiter_ceiling_db": limiter_ceiling_db(args),
            "split_audio": args.split_audio,
        },
        "inputs": hashes({
            "song": song,
            "vocals": stems.vocals,
            "instrumental": stems.instrumental,
            "reference_vocals": reference.vocals if reference else None,
            "pth": model.pth,
            "index": model.index,
        }),
        "separation": stems.origin,
        "transpose_estimate": estimate,
        "applio_command": format_command(command),
        "mix": mix_stats,
        "result": result,
        "timings": timings,
    }
    sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.keep_intermediates:
        shutil.rmtree(work, ignore_errors=True)
    print(f"cover: {record_path(output)} | {result['duration_s']:.2f}s (song {song_duration:.2f}s) | "
          f"{result['loudness_lufs']:.1f} LUFS, peak {result['peak_dbfs']:.2f} dBFS | "
          f"vocal lag {mix_stats['vocal_lag_ms']:+.0f} ms | {timings['total_s']:.0f}s total")
    print(f"settings: {record_path(sidecar)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
