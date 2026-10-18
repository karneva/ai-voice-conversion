#!/usr/bin/env python3
"""Score converted vocals and upsert aggregate rows into results/runs.csv.

Subcommands:
  calibrate    target centroid, similarity ceiling/floor, source-clip baselines
  run          score runs/<run_id>/infer/<set>/<epoch>e/<combo>/*.wav
  source-only  score the unconverted clips as if they were outputs ('_source' rows)

Metrics are computed one model at a time over all clips (speaker -> pitch ->
ASR -> [UTMOS -> SingMOS] -> MCD) so only one model sits on the GPU at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from eval_lib import asr, audio, mcd, mos, pitch, speaker
from eval_lib.cache import JsonCache
from eval_lib.gpu import StageMonitor, pick_device

EVAL_VERSION = "1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
SETS = ("source_eval", "heldout")
SOURCE_RUN_ID = "_source"
# Written by run_experiment.py in every infer/<set>/<epoch>e/<combo>/: clip_id -> {sha256, transpose} of the source clip.
PROVENANCE_FILE = "_sources.json"
PITCH_COLUMNS = ("f0_rmse_cents", "gpe50", "octave_err", "fpc")

RUNS_COLUMNS = [
    "run_id", "track", "set", "epoch", "step",
    "f0_method", "index_rate", "protect", "volume_envelope",
    "n_clips", "n_pitch_clips",
    "sim_target", "sim_target_norm", "sim_source",
    "f0_rmse_cents", "gpe50", "octave_err", "vuv_err", "fpc",
    "cer", "cer_source", "dcer", "cer_ref",
    "utmos", "utmos_source", "singmos", "singmos_source",
    "mcd",
    "eval_version",
]
# Contract key plus track, so '_source' rows of different tracks never overwrite each other.
UPSERT_KEY = ("run_id", "track", "set", "epoch", "f0_method", "index_rate", "protect", "volume_envelope")

CLIP_COLUMNS = [
    "clip_id", "output_path", "reference_path", "transpose", "output_sha256", "reference_sha256", "active_s",
    "sim_target", "sim_source",
    "f0_rmse_cents", "gpe50", "octave_err", "vuv_err", "fpc",
    "voiced_frames_ref", "voiced_frames_out", "voiced_frames_both",
    "cer", "cer_source", "ref_kind", "ref_chars",
    "utmos", "utmos_source", "singmos", "singmos_source",
    "mcd",
]
# Transcripts can hold song lyrics, so they live in a gitignored sidecar (<combo>.text.csv), not the metric CSV.
TEXT_COLUMNS = ["clip_id", "ref_text", "asr_text", "source_text"]
MEAN_COLUMNS = ("sim_target", "sim_source", "f0_rmse_cents", "gpe50", "octave_err", "vuv_err", "fpc", "mcd")

COMBO_RE = re.compile(
    r"^f0-(?P<f0_method>.+)_ir(?P<index_rate>\d+(?:\.\d+)?)_p(?P<protect>\d+(?:\.\d+)?)"
    r"_ve(?P<volume_envelope>\d+(?:\.\d+)?)$"
)
EPOCH_DIR_RE = re.compile(r"^(\d+)e$")


# ---------------------------------------------------------------- data model


@dataclass(frozen=True)
class Workspace:
    root: Path
    dataset: Path
    runs: Path
    results: Path

    @property
    def cache(self) -> Path:
        return self.results / "cache"

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path


@dataclass(frozen=True)
class Clip:
    clip_id: str
    path: Path
    singer: str
    transpose: int
    lyrics: str
    sha256: str


@dataclass
class Pair:
    clip: Clip
    output: Path
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Group:
    run_id: str
    track: str
    set_name: str
    epoch: str
    step: str
    combo: str
    csv_path: Path
    pairs: list[Pair]

    def label(self) -> str:
        parts = [self.run_id, self.set_name, f"{self.epoch}e" if self.epoch else "", self.combo]
        return "/".join(p for p in parts if p)


# ---------------------------------------------------------------- helpers


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def is_missing(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def to_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def fmt(value: Any) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def nan_mean(values: Iterable[Any]) -> float | None:
    numbers = [v for v in (to_float(x) for x in values) if v is not None]
    return float(np.mean(numbers)) if numbers else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: fmt(row.get(col)) for col in columns})
    os.replace(tmp, path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_json(obj: Any) -> Iterator[tuple[str | None, Any]]:
    """Yield (key, value) for every node of a JSON-like object."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from iter_json(value)
    elif isinstance(obj, list):
        for value in obj:
            yield None, value
            yield from iter_json(value)


def memoize(fn: Callable[[Path], Any]) -> Callable[[Path], Any]:
    store: dict[Path, Any] = {}

    def wrapper(path: Path) -> Any:
        if path not in store:
            store[path] = fn(path)
        return store[path]

    return wrapper


file_sha256 = memoize(audio.sha256_file)


# ---------------------------------------------------------------- dataset / runs


def read_clips(ws: Workspace, track: str, set_name: str, base: Path | None = None) -> dict[str, Clip]:
    csv_path = (base or ws.dataset / track) / set_name / "clips.csv"
    if not csv_path.exists():
        fail(f"clip list not found: {csv_path}")
    clips: dict[str, Clip] = {}
    for row in read_csv(csv_path):
        clip_id = row["clip_id"].strip()
        if clip_id in clips:
            fail(f"duplicate clip_id {clip_id} in {csv_path}")
        path = ws.resolve(row["path"].strip())
        if not path.exists():
            fail(f"clip audio not found: {path} ({csv_path})")
        transpose = row.get("transpose", "").strip()
        clips[clip_id] = Clip(
            clip_id=clip_id,
            path=path,
            singer=row.get("singer", "").strip(),
            transpose=int(float(transpose)) if transpose and set_name != "heldout" else 0,
            lyrics=row.get("lyrics", "") or "",
            sha256=file_sha256(path),
        )
    if not clips:
        fail(f"no clips in {csv_path}")
    return clips


def parse_combo(tag: str) -> dict[str, str]:
    match = COMBO_RE.match(tag)
    if not match:
        raise ValueError(f"combo tag does not match f0-<method>_ir<x>_p<y>_ve<z>: {tag}")
    return match.groupdict()


def load_run_info(ws: Workspace, run_id: str) -> dict[str, Any]:
    path = ws.runs / run_id / "run.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_track(run_info: dict[str, Any]) -> str | None:
    for key, value in iter_json(run_info):
        if key == "track" and isinstance(value, str) and value:
            return value
    return None


def find_step(ws: Workspace, run_id: str, epoch: str, run_info: dict[str, Any]) -> str:
    pattern = re.compile(rf"_{int(epoch)}e_(\d+)s\.pth$")
    for _, value in iter_json(run_info):
        if isinstance(value, str):
            match = pattern.search(value)
            if match:
                return match.group(1)
    logs_dir = ws.root / "Applio" / "logs" / run_id
    if logs_dir.is_dir():
        for weight in logs_dir.glob(f"{run_id}_{int(epoch)}e_*s.pth"):
            match = pattern.search(weight.name)
            if match:
                return match.group(1)
    return ""


def find_eval_dir(ws: Workspace, track: str, run_info: dict[str, Any]) -> Path:
    """Clip-list folder the run was inferred from (run_experiment.py's eval_dir, else dataset/<track>)."""
    default = ws.dataset / track
    for key, value in iter_json(run_info):
        if key == "eval_dir" and isinstance(value, str) and value:
            path = ws.resolve(value).resolve()
            if path != default.resolve():
                log(f"warning: run used eval_dir {value}; similarity calibration still uses {ws.rel(default)}")
            return path
    return default


def parse_epochs(value: str) -> set[int] | None:
    if value == "all":
        return None
    return {int(v.strip().rstrip("e")) for v in value.split(",") if v.strip()}


def parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def preview(items: list[str], limit: int = 5) -> str:
    return ", ".join(items[:limit]) + (" ..." if len(items) > limit else "")


def read_provenance(combo_dir: Path) -> dict[str, Any] | None:
    path = combo_dir / PROVENANCE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def match_outputs(combo_dir: Path, clips: dict[str, Clip], label: str) -> list[Pair]:
    """Pair outputs with clips by clip_id; refuse outputs converted from other audio or another transpose."""
    provenance = read_provenance(combo_dir)
    pairs, stale, unverified = [], [], []
    for wav in sorted(combo_dir.glob("*.wav")):
        clip = clips.get(wav.stem)
        if clip is None:
            log(f"warning: {label}: no clip_id matches output {wav.name}; skipped")
            continue
        entry = (provenance or {}).get(clip.clip_id)
        if entry is None:
            unverified.append(clip.clip_id)
        elif entry.get("sha256") != clip.sha256 or int(entry.get("transpose", 0)) != clip.transpose:
            stale.append(clip.clip_id)
        pairs.append(Pair(clip=clip, output=wav))
    if stale:
        fail(
            f"{label}: {len(stale)} output(s) were converted from a different clip audio or transpose than the "
            f"current clips.csv: {preview(stale)}. Re-run run_experiment.py infer (it re-converts stale outputs)."
        )
    if unverified:
        log(f"warning: {label}: {len(unverified)} output(s) have no {PROVENANCE_FILE} entry, so their source clip "
            f"cannot be verified: {preview(unverified)}")
    missing = sorted(set(clips) - {p.clip.clip_id for p in pairs})
    if missing:
        log(f"warning: {label}: {len(missing)} clip(s) have no output: {preview(missing)}")
    return pairs


def discover_groups(
    ws: Workspace,
    run_id: str,
    track: str,
    sets: list[str],
    epochs: set[int] | None,
    combos: list[str] | None,
    run_info: dict[str, Any],
) -> list[Group]:
    groups: list[Group] = []
    run_dir = ws.runs / run_id
    eval_dir = find_eval_dir(ws, track, run_info)
    for set_name in sets:
        base = run_dir / "infer" / set_name
        if not base.is_dir():
            log(f"warning: no outputs for set {set_name}: {base}")
            continue
        clips = read_clips(ws, track, set_name, eval_dir)
        epoch_dirs = [d for d in base.iterdir() if d.is_dir() and EPOCH_DIR_RE.match(d.name)]
        for epoch_dir in sorted(epoch_dirs, key=lambda d: int(EPOCH_DIR_RE.match(d.name).group(1))):
            epoch = EPOCH_DIR_RE.match(epoch_dir.name).group(1)
            if epochs is not None and int(epoch) not in epochs:
                continue
            for combo_dir in sorted(d for d in epoch_dir.iterdir() if d.is_dir()):
                if combos is not None and combo_dir.name not in combos:
                    continue
                if not COMBO_RE.match(combo_dir.name):
                    log(f"warning: skipping folder with an invalid combo tag: {combo_dir}")
                    continue
                label = f"{run_id}/{set_name}/{epoch_dir.name}/{combo_dir.name}"
                pairs = match_outputs(combo_dir, clips, label)
                if not pairs:
                    continue
                groups.append(
                    Group(
                        run_id=run_id,
                        track=track,
                        set_name=set_name,
                        epoch=epoch,
                        step=find_step(ws, run_id, epoch, run_info),
                        combo=combo_dir.name,
                        csv_path=run_dir / "metrics" / set_name / epoch_dir.name / f"{combo_dir.name}.csv",
                        pairs=pairs,
                    )
                )
    return groups


# ---------------------------------------------------------------- calibration


def calibration_path(ws: Workspace, track: str) -> Path:
    return ws.results / f"calibration_{track}.json"


def clip_hashes(clips: dict[str, Clip]) -> dict[str, str]:
    return {cid: clips[cid].sha256 for cid in sorted(clips)}


def load_calibration(ws: Workspace, track: str, targets: dict[str, Clip]) -> dict[str, Any]:
    """The calibration must have been built from exactly the current held-out and source_eval clip audio."""
    path = calibration_path(ws, track)
    rerun = f"Re-run: evaluate.py calibrate --track {track}"
    if not path.exists():
        fail(f"calibration file not found: {path}. {rerun}")
    calib = json.loads(path.read_text(encoding="utf-8"))
    hashes = calib.get("clip_sha256")
    if not hashes:
        fail(f"{path} has no per-clip hashes (made before eval_version 1.1.0). {rerun}")
    sources = read_clips(ws, track, "source_eval")
    for set_name, clips in (("heldout", targets), ("source_eval", sources)):
        if hashes.get(set_name) != clip_hashes(clips):
            fail(f"{set_name} clips (ids or audio) changed since calibration ({path}). {rerun}")
    if calib.get("eval_version") != EVAL_VERSION:
        log(f"warning: calibration was made with eval_version {calib.get('eval_version')}, now {EVAL_VERSION}")
    return calib


def normalized_sim(value: float | None, calib: dict[str, Any]) -> float | None:
    if value is None:
        return None
    span = calib["sim_ceiling"] - calib["sim_floor"]
    return (value - calib["sim_floor"]) / span if span > 0 else None


# ---------------------------------------------------------------- metric stages


def stage_speaker(ws: Workspace, groups: list[Group], targets: dict[str, Clip], device: str) -> dict[str, Any]:
    with StageMonitor("speaker", device) as mon:
        encoder = speaker.SpeakerEncoder(ws.cache / "models", device)
        try:
            def embed(path: Path) -> tuple[np.ndarray, float]:
                mon.tick()
                wav = audio.load_16k(path)
                return encoder.embed(wav), audio.active_seconds(wav)

            embed_cached = memoize(embed)
            target_embs = {cid: embed_cached(clip.path)[0] for cid, clip in targets.items()}
            full_centroid = speaker.centroid(target_embs.values())
            for group in groups:
                for pair in group.pairs:
                    out_emb, pair.metrics["active_s"] = embed_cached(pair.output)
                    if group.set_name == "heldout" and pair.clip.clip_id in target_embs:
                        others = (e for k, e in target_embs.items() if k != pair.clip.clip_id)
                        reference = speaker.centroid(others)
                    else:
                        reference = full_centroid
                    pair.metrics["sim_target"] = speaker.cosine(out_emb, reference)
                    if group.set_name == "source_eval":
                        pair.metrics["sim_source"] = speaker.cosine(out_emb, embed_cached(pair.clip.path)[0])
        finally:
            encoder.close()
    return mon.stats


def stage_pitch(groups: list[Group], device: str) -> dict[str, Any]:
    with StageMonitor("pitch", device) as mon:
        extractor = pitch.PitchExtractor(device)
        try:
            def extract(path: Path) -> pitch.F0Track:
                mon.tick()
                return extractor.extract(audio.load_16k(path))

            extract_cached = memoize(extract)
            for group in groups:
                for pair in group.pairs:
                    reference = extract_cached(pair.clip.path)
                    output = extract_cached(pair.output)
                    pair.metrics.update(pitch.pitch_metrics(reference, output, pair.clip.transpose))
        finally:
            extractor.close()
    return mon.stats


class LazyModel:
    """Loads a model on first use so fully cached stages never touch the GPU."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.model: Any = None

    def get(self) -> Any:
        if self.model is None:
            self.model = self.factory()
        return self.model

    def close(self) -> None:
        if self.model is not None:
            self.model.close()
            self.model = None


def asr_cache(ws: Workspace) -> JsonCache:
    tag = asr.cache_tag()
    meta = {"model": asr.MODEL_NAME, "compute_type": asr.COMPUTE_TYPE, "decode": asr.DECODE_OPTIONS}
    return JsonCache(ws.cache / "asr" / f"{tag}.json", meta=meta)


def transcribe_files(ws: Workspace, paths: Iterable[Path], device: str, mon: StageMonitor) -> Callable[[Path], str]:
    """Return a lookup path -> transcript; transcripts are cached by file hash."""
    cache = asr_cache(ws)
    model = LazyModel(lambda: asr.Transcriber(device))
    texts: dict[Path, str] = {}
    try:
        for path in dict.fromkeys(paths):
            key = audio.sha256_file(path)
            cached = cache.get(key)
            if cached is None:
                text = model.get().transcribe(audio.load_16k(path))
                cache.set(key, {"path": ws.rel(path), "text": text})
                mon.tick()
                if len(cache.entries) % 25 == 0:
                    cache.save()
            else:
                text = cached["text"]
            texts[path] = text
    finally:
        model.close()
        cache.save()
    return texts.__getitem__


def stage_asr(ws: Workspace, groups: list[Group], device: str) -> dict[str, Any]:
    with StageMonitor("asr", device) as mon:
        paths = [p for g in groups for pair in g.pairs for p in (pair.clip.path, pair.output)]
        text_of = transcribe_files(ws, paths, device, mon)
        for group in groups:
            for pair in group.pairs:
                hyp = text_of(pair.output)
                src = text_of(pair.clip.path)
                has_lyrics = bool(asr.normalize_text(pair.clip.lyrics))
                ref = pair.clip.lyrics if has_lyrics else src
                pair.metrics.update(
                    ref_kind="lyrics" if has_lyrics else "asr",
                    ref_chars=len(asr.normalize_text(ref)),
                    ref_text=ref,
                    asr_text=hyp,
                    source_text=src,
                    cer=asr.cer(ref, hyp),
                    cer_source=asr.cer(ref, src),
                )
    return mon.stats


def score_files(
    ws: Workspace, name: str, paths: Iterable[Path], device: str, mon: StageMonitor | None, cache_only: bool = False
) -> Callable[[Path], float | None]:
    """Return a lookup path -> score; with cache_only, uncached files map to None and no model loads."""
    repo, entry = mos.PREDICTORS[name]
    cache = JsonCache(ws.cache / "mos" / f"{entry}.json", meta={"repo": repo, "entry": entry})
    model = LazyModel(lambda: mos.MosPredictor(name, device))
    scores: dict[Path, float | None] = {}
    try:
        for path in dict.fromkeys(paths):
            key = audio.sha256_file(path)
            if key not in cache and not cache_only:
                cache.set(key, {"path": ws.rel(path), "score": model.get().score(audio.load_16k(path))})
                if mon is not None:
                    mon.tick()
            entry_value = cache.get(key)
            scores[path] = entry_value["score"] if entry_value else None
    finally:
        model.close()
        cache.save()
    return scores.__getitem__


def fill_mos(ws: Workspace, groups: list[Group], name: str, device: str, mon: StageMonitor | None) -> None:
    paths = [p for g in groups for pair in g.pairs for p in (pair.clip.path, pair.output)]
    score_of = score_files(ws, name, paths, device, mon, cache_only=mon is None)
    for group in groups:
        for pair in group.pairs:
            pair.metrics[name] = score_of(pair.output)
            pair.metrics[f"{name}_source"] = score_of(pair.clip.path)


def stage_mos(ws: Workspace, groups: list[Group], name: str, device: str) -> dict[str, Any]:
    with StageMonitor(name, device) as mon:
        fill_mos(ws, groups, name, device, mon)
    return mon.stats


def stage_mcd(groups: list[Group]) -> dict[str, Any]:
    with StageMonitor("mcd", "cpu") as mon:
        for group in groups:
            if group.set_name != "heldout":
                continue
            for pair in group.pairs:
                reference = audio.load_16k(pair.clip.path)
                output = audio.load_16k(pair.output)
                pair.metrics["mcd"] = mcd.mel_cepstral_distance(reference, output)
                mon.tick()
    return mon.stats


def evaluate_groups(
    ws: Workspace, groups: list[Group], targets: dict[str, Clip], device: str, with_mos: bool
) -> list[dict[str, Any]]:
    stats = [stage_speaker(ws, groups, targets, device), stage_pitch(groups, device), stage_asr(ws, groups, device)]
    for name in ("utmos", "singmos"):
        if with_mos:
            stats.append(stage_mos(ws, groups, name, device))
        else:
            fill_mos(ws, groups, name, device, None)  # cached scores only, no model load
    stats.append(stage_mcd(groups))
    return stats


# ---------------------------------------------------------------- aggregation / output


def clip_rows(ws: Workspace, group: Group) -> list[dict[str, Any]]:
    rows = []
    for pair in group.pairs:
        row = {
            "clip_id": pair.clip.clip_id,
            "output_path": ws.rel(pair.output),
            "reference_path": ws.rel(pair.clip.path),
            "transpose": pair.clip.transpose,
            "output_sha256": file_sha256(pair.output),
            "reference_sha256": pair.clip.sha256,
        }
        row.update(pair.metrics)
        rows.append(row)
    return rows


def aggregate(group: Group, rows: list[dict[str, Any]], calib: dict[str, Any]) -> dict[str, Any]:
    combo = parse_combo(group.combo) if group.combo else {}
    result: dict[str, Any] = {
        "run_id": group.run_id,
        "track": group.track,
        "set": group.set_name,
        "epoch": group.epoch,
        "step": group.step,
        "f0_method": combo.get("f0_method", ""),
        "index_rate": combo.get("index_rate", ""),
        "protect": combo.get("protect", ""),
        "volume_envelope": combo.get("volume_envelope", ""),
        "n_clips": len(rows),
        "eval_version": EVAL_VERSION,
    }
    for column in MEAN_COLUMNS:
        result[column] = nan_mean(row.get(column) for row in rows)
    if group.set_name == "heldout":
        result["sim_source"] = None
    result["sim_target_norm"] = normalized_sim(result["sim_target"], calib)
    # Pitch means skip clips without jointly voiced frames (e.g. a silent output); count what they cover.
    result["n_pitch_clips"] = sum(to_float(row.get("f0_rmse_cents")) is not None for row in rows)
    if result["n_pitch_clips"] < len(rows):
        log(f"warning: {group.label()}: pitch metrics cover only {result['n_pitch_clips']}/{len(rows)} clip(s); "
            "the others have fewer than 2 frames voiced in both reference and output")

    kinds = sorted({row.get("ref_kind") for row in rows if row.get("ref_kind")})
    result["cer_ref"] = kinds[0] if len(kinds) == 1 else ("mixed" if kinds else "")
    refs = [row.get("ref_text") or "" for row in rows]
    result["cer"] = asr.corpus_cer(refs, [row.get("asr_text") or "" for row in rows])
    result["cer_source"] = asr.corpus_cer(refs, [row.get("source_text") or "" for row in rows])
    if result["cer"] is not None and result["cer_source"] is not None:
        result["dcer"] = result["cer"] - result["cer_source"]
    for name in ("utmos", "singmos"):
        result[name] = nan_mean(row.get(name) for row in rows)
        result[f"{name}_source"] = nan_mean(row.get(f"{name}_source") for row in rows)
    return result


def upsert_runs(ws: Workspace, new_rows: list[dict[str, Any]]) -> Path:
    path = ws.results / "runs.csv"
    existing = read_csv(path) if path.exists() else []
    if existing and set(existing[0]) - set(RUNS_COLUMNS):
        extra = sorted(set(existing[0]) - set(RUNS_COLUMNS))
        log(f"warning: dropping unknown runs.csv column(s): {', '.join(extra)}")

    def key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(fmt(row.get(col)) for col in UPSERT_KEY)

    index = {key(row): i for i, row in enumerate(existing)}
    rows: list[dict[str, Any]] = list(existing)
    for row in new_rows:
        formatted = {col: fmt(row.get(col)) for col in RUNS_COLUMNS}
        position = index.get(key(formatted))
        if position is None:
            index[key(formatted)] = len(rows)
            rows.append(formatted)
        else:
            rows[position] = formatted
    write_csv(path, RUNS_COLUMNS, rows)
    return path


def text_csv_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}.text.csv")


def write_clip_csvs(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(csv_path, CLIP_COLUMNS, rows)
    write_csv(text_csv_path(csv_path), TEXT_COLUMNS, rows)


def read_clip_rows(csv_path: Path) -> list[dict[str, str]]:
    """Per-clip metrics joined with their transcripts (older CSVs carry the text columns inline)."""
    rows = read_csv(csv_path)
    texts_path = text_csv_path(csv_path)
    if texts_path.exists():
        texts = {row["clip_id"]: row for row in read_csv(texts_path)}
        for row in rows:
            row.update({k: v for k, v in texts.get(row["clip_id"], {}).items() if k != "clip_id"})
    return rows


def has_complete_csv(group: Group, with_mos: bool) -> bool:
    """A per-clip CSV is reusable only if it scored exactly these output and reference files."""
    if not group.csv_path.exists():
        return False
    if not text_csv_path(group.csv_path).exists() and "asr_text" not in read_csv(group.csv_path)[0]:
        return False  # transcripts are needed to recompute corpus CER
    rows = read_csv(group.csv_path)
    if {r["clip_id"] for r in rows} != {p.clip.clip_id for p in group.pairs}:
        return False
    expected = {
        p.clip.clip_id: (file_sha256(p.output), p.clip.sha256, str(p.clip.transpose)) for p in group.pairs
    }
    for row in rows:
        if (row.get("output_sha256"), row.get("reference_sha256"), row.get("transpose")) != expected[row["clip_id"]]:
            return False
    return not with_mos or all(r.get("utmos") and r.get("singmos") for r in rows)


def finish_groups(
    ws: Workspace,
    groups: list[Group],
    calib: dict[str, Any],
    stats: list[dict[str, Any]],
    log_dir: Path,
    reused: list[Group],
) -> None:
    aggregates = []
    for group in groups:
        rows = clip_rows(ws, group)
        write_clip_csvs(group.csv_path, rows)
        aggregates.append(aggregate(group, rows, calib))
        log(f"wrote {ws.rel(group.csv_path)} ({len(rows)} clip(s))")
    for group in reused:
        aggregates.append(aggregate(group, read_clip_rows(group.csv_path), calib))
        log(f"reused {ws.rel(group.csv_path)}")
    if not aggregates:
        log("nothing to write")
        return
    runs_csv = upsert_runs(ws, aggregates)
    log(f"upserted {len(aggregates)} row(s) into {ws.rel(runs_csv)}")
    entry = {
        "timestamp": now_iso(),
        "eval_version": EVAL_VERSION,
        "argv": sys.argv[1:],
        "groups": [g.label() for g in groups],
        "reused": [g.label() for g in reused],
        "clips": sum(len(g.pairs) for g in groups),
        "stages": stats,
        "pitch_config": pitch.PitchConfig().as_dict(),
        "mcd_config": mcd.McdConfig().as_dict(),
        "asr": {"model": asr.MODEL_NAME, "compute_type": asr.COMPUTE_TYPE, **asr.DECODE_OPTIONS},
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "eval_log.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- subcommands


def cmd_calibrate(args: argparse.Namespace, ws: Workspace) -> None:
    device = pick_device(args.device)
    targets = read_clips(ws, args.track, "heldout")
    sources = read_clips(ws, args.track, "source_eval")
    if len(targets) < 2:
        fail("calibration needs at least 2 held-out clips")
    log(f"calibrate {args.track}: {len(targets)} held-out clip(s), {len(sources)} source clip(s), device {device}")

    stats = []
    with StageMonitor("speaker", device) as mon:
        encoder = speaker.SpeakerEncoder(ws.cache / "models", device)
        try:
            target_embs = {cid: encoder.embed(audio.load_16k(c.path)) for cid, c in targets.items()}
            source_embs = {cid: encoder.embed(audio.load_16k(c.path)) for cid, c in sources.items()}
            mon.tick(len(target_embs) + len(source_embs))
        finally:
            encoder.close()
    stats.append(mon.stats)

    loo = speaker.leave_one_out_cosines(target_embs)
    full_centroid = speaker.centroid(target_embs.values())
    floor = {cid: speaker.cosine(e, full_centroid) for cid, e in source_embs.items()}
    if np.mean(list(loo.values())) <= np.mean(list(floor.values())):
        log("warning: similarity ceiling <= floor; sim_target_norm will be meaningless")

    all_paths = [c.path for c in targets.values()] + [c.path for c in sources.values()]
    with StageMonitor("asr", device) as mon:
        transcribe_files(ws, all_paths, device, mon)
    stats.append(mon.stats)
    if args.mos:
        for name in ("utmos", "singmos"):
            with StageMonitor(name, device) as mon:
                score_files(ws, name, all_paths, device, mon)
            stats.append(mon.stats)

    calib = {
        "track": args.track,
        "eval_version": EVAL_VERSION,
        "created_at": now_iso(),
        "speaker_model": speaker.MODEL_ID,
        "asr_model": asr.MODEL_NAME,
        "centroid_clips": sorted(targets),
        "source_clips": sorted(sources),
        "clip_sha256": {"heldout": clip_hashes(targets), "source_eval": clip_hashes(sources)},
        "n_target": len(targets),
        "n_source": len(sources),
        "sim_ceiling": float(np.mean(list(loo.values()))),
        "sim_ceiling_std": float(np.std(list(loo.values()))),
        "sim_floor": float(np.mean(list(floor.values()))),
        "sim_floor_std": float(np.std(list(floor.values()))),
        "ceiling_method": "mean cosine of each held-out clip vs the centroid of the other held-out clips",
        "floor_method": "mean cosine of each source_eval clip vs the centroid of all held-out clips",
        "per_clip": {
            "target_loo": {k: round(v, 6) for k, v in sorted(loo.items())},
            "source_vs_centroid": {k: round(v, 6) for k, v in sorted(floor.items())},
        },
        "heldout_clips_csv": ws.rel(ws.dataset / args.track / "heldout" / "clips.csv"),
        "source_clips_csv": ws.rel(ws.dataset / args.track / "source_eval" / "clips.csv"),
        "stages": stats,
    }
    path = calibration_path(ws, args.track)
    write_json(path, calib)
    log(
        f"sim ceiling {calib['sim_ceiling']:.4f} (sd {calib['sim_ceiling_std']:.4f}), "
        f"floor {calib['sim_floor']:.4f} (sd {calib['sim_floor_std']:.4f})"
    )
    log(f"wrote {ws.rel(path)}")


def run_groups(ws: Workspace, args: argparse.Namespace, groups: list[Group], targets: dict[str, Clip],
               calib: dict[str, Any], log_dir: Path) -> None:
    reused: list[Group] = []
    if args.skip_existing:
        reused = [g for g in groups if has_complete_csv(g, args.mos)]
        groups = [g for g in groups if g not in reused]
    device = pick_device(args.device)
    n_clips = sum(len(g.pairs) for g in groups)
    log(f"evaluating {len(groups)} group(s), {n_clips} output clip(s) on {device}; {len(reused)} reused")
    stats = evaluate_groups(ws, groups, targets, device, args.mos) if groups else []
    finish_groups(ws, groups, calib, stats, log_dir, reused)


def cmd_run(args: argparse.Namespace, ws: Workspace) -> None:
    run_dir = ws.runs / args.run
    if not run_dir.is_dir():
        fail(f"run folder not found: {run_dir}")
    run_info = load_run_info(ws, args.run)
    track = args.track or find_track(run_info)
    if not track:
        fail(f"track not found in {run_dir / 'run.json'}; pass --track")
    targets = read_clips(ws, track, "heldout")
    calib = load_calibration(ws, track, targets)
    sets = parse_list(args.sets)
    unknown = set(sets) - set(SETS)
    if unknown:
        fail(f"unknown set(s): {', '.join(sorted(unknown))}")
    combos = None if args.combos == "all" else parse_list(args.combos)
    groups = discover_groups(ws, args.run, track, sets, parse_epochs(args.epochs), combos, run_info)
    if not groups:
        fail(f"no outputs found under {run_dir / 'infer'} for the given filters")
    run_groups(ws, args, groups, targets, calib, run_dir / "metrics")


def cmd_source_only(args: argparse.Namespace, ws: Workspace) -> None:
    targets = read_clips(ws, args.track, "heldout")
    calib = load_calibration(ws, args.track, targets)
    out_dir = ws.runs / SOURCE_RUN_ID / "metrics" / args.track
    groups = []
    for set_name in parse_list(args.sets):
        if set_name not in SETS:
            fail(f"unknown set: {set_name}")
        clips = targets if set_name == "heldout" else read_clips(ws, args.track, set_name)
        # The unconverted clip carries no transpose, so it is compared to itself at 0 semitones.
        pairs = [Pair(clip=replace(c, transpose=0), output=c.path) for c in clips.values()]
        groups.append(
            Group(SOURCE_RUN_ID, args.track, set_name, "", "", "", out_dir / f"{set_name}.csv", pairs)
        )
    run_groups(ws, args, groups, targets, calib, out_dir)


# ---------------------------------------------------------------- CLI


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Base for relative paths. Default: repo root.")
    parser.add_argument("--dataset-root", type=Path, help="Default: <root>/dataset.")
    parser.add_argument("--runs-dir", type=Path, help="Default: <root>/runs.")
    parser.add_argument("--results-dir", type=Path, help="Default: <root>/results (runs.csv, calibration, cache).")
    parser.add_argument("--device", help="torch device, e.g. cuda or cpu. Default: cuda if available.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="Build the target centroid and similarity ceiling/floor.")
    calibrate.add_argument("--track", required=True, choices=["private", "public"])
    calibrate.add_argument("--mos", action="store_true", help="Also cache UTMOS22/SingMOS-Pro source scores.")
    add_common(calibrate)

    run = sub.add_parser("run", help="Score converted outputs of one run.")
    run.add_argument("--run", required=True, help="Run ID (folder under runs/).")
    run.add_argument("--sets", default=",".join(SETS), help="Comma list of sets. Default: source_eval,heldout.")
    run.add_argument("--epochs", default="all", help="Comma list like 10,20 or 'all'. Default: all.")
    run.add_argument("--combos", default="all", help="Comma list of combo tags or 'all'. Default: all.")
    run.add_argument("--track", help="Override the track from run.json.")
    run.add_argument("--mos", action="store_true", help="Also compute UTMOS22 and SingMOS-Pro.")
    run.add_argument("--skip-existing", action="store_true", help="Reuse per-clip CSVs that already cover every clip.")
    add_common(run)

    source = sub.add_parser("source-only", help="Score unconverted clips as a '_source' reference row.")
    source.add_argument("--track", required=True, choices=["private", "public"])
    source.add_argument("--sets", default=",".join(SETS), help="Comma list of sets. Default: source_eval,heldout.")
    source.add_argument("--mos", action="store_true", help="Also compute UTMOS22 and SingMOS-Pro.")
    source.add_argument("--skip-existing", action="store_true", help="Reuse per-clip CSVs that already cover every clip.")
    add_common(source)
    return parser.parse_args()


def make_workspace(args: argparse.Namespace) -> Workspace:
    root = args.root.resolve()

    def under(value: Path | None, name: str) -> Path:
        if value is None:
            return root / name
        return value.resolve() if value.is_absolute() else (root / value).resolve()

    return Workspace(
        root=root,
        dataset=under(args.dataset_root, "dataset"),
        runs=under(args.runs_dir, "runs"),
        results=under(args.results_dir, "results"),
    )


def main() -> None:
    args = parse_args()
    ws = make_workspace(args)
    commands = {"calibrate": cmd_calibrate, "run": cmd_run, "source-only": cmd_source_only}
    commands[args.command](args, ws)


if __name__ == "__main__":
    main()
