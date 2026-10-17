#!/usr/bin/env python3
"""Run one RVC experiment (Applio 3.6.6) from a JSON config and record everything under runs/<run_id>/.

Steps: prepare (preprocess + extract), train (+ index), infer (batch-infer over eval sets).

Launch strategy: Applio's core.py starts preprocess/extract/train/index through run_quiet()
(CREATE_NO_WINDOW). When core.py's stdout is a pipe or file, the grandchild's output is lost on
Windows (verified), so training logs could not be parsed or saved. This runner therefore calls
the same scripts core.py calls (rvc/train/preprocess/preprocess.py, rvc/train/extract/extract.py,
rvc/train/train.py, rvc/train/process/extract_index.py) with exactly the positional argv that
core.py builds, cwd=Applio and core.py's Windows environment variables. The equivalent core.py
command is written next to every real command in commands.log. Inference runs in-process inside
core.py (no grandchild), so `core.py batch-infer` is called as-is.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
APPLIO_DIR = REPO_ROOT / "Applio"
APPLIO_PY = APPLIO_DIR / "env" / "python.exe"
APPLIO_LOGS = APPLIO_DIR / "logs"
RUNS_DIR = REPO_ROOT / "runs"

STEPS = ("prepare", "train", "infer")
EVAL_SETS = ("source_eval", "heldout")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
CHECKPOINT_STEP = re.compile(r"_\d+e_(\d+)s\.pth$")
EPOCH_LINE = re.compile(
    r"^(?P<model>\S+) \| epoch=(?P<epoch>\d+) \| step=(?P<step>\d+) \| "
    r"time=(?P<clock>\d{2}:\d{2}:\d{2}) \| training_speed=(?P<speed>\d+:\d{2}:\d{2})"
)
VRAM_SAMPLE_SECONDS = 5.0

TRACEBACK = r"Traceback \(most recent call last\)"
FAILURE_PATTERNS: dict[str, list[str]] = {
    "preprocess": [
        TRACEBACK,
        r"Error processing audio",
        r"The dataset path (does not exist|is not a directory)",
        r"No audio files found",
        r"Speaker ID folder is expected",
    ],
    "extract": [
        TRACEBACK,
        r"An error occurred extracting file",
        r"\[!\] Warning",
        r"files failed pitch extraction",
        r"produced NaN values",
        r"^ERROR:",
        r"not found at .*Did you run the preprocessing step",
    ],
    "train": [
        TRACEBACK,
        r"^Error:",
        r"Config file not found",
        r"No wav file found",
        r"Not enough data present",
        r"do not match the selected model",
        r"out of memory",
        r"Training worker process failed",
    ],
    "index": [TRACEBACK, r"not found at", r"could not be loaded"],
    "infer": [TRACEBACK, r"An error occurred", r"Batch conversion failed"],
}
# Per-file extract errors that Applio survives by leaving the slice out of filelist.txt
# (RMVPE raises on slices shorter than ~0.2 s). Allowed up to EXTRACT_MAX_FAILED_FRACTION of slices.
TOLERATED_PATTERNS: dict[str, list[str]] = {
    "extract": [r"An error occurred extracting file", r"\d+/\d+ files failed pitch extraction"],
}
EXTRACT_FAILED_FILE = re.compile(r"An error occurred extracting file (.+?) on \S+: ")
EXTRACT_FAILED_LINE = re.compile(r"(\d+)/(\d+) files failed pitch extraction")
EXTRACT_MAX_FAILED_FRACTION = 0.01

# Applio's core.py sets these on Windows before spawning its scripts; children inherit them.
APPLIO_ENV_DEFAULTS = {
    "MIOPEN_FIND_MODE": "2",
    "MIOPEN_DEBUG_DISABLE_FIND_DB": "1",
    "MIOPEN_LOG_LEVEL": "0",
    "MIOPEN_ENABLE_LOGGING": "0",
    "DISABLE_ADDMM_CUDA_LT": "1",
    "AMD_COMGR_CACHE": "0",
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0",
}

# Step-record keys that accumulate across invocations (training resumes); all others are reset per invocation.
ACCUMULATED_KEYS = ("epochs", "peak_vram_mib", "vram_samples")

# Train-step keys measured by the last real training; kept when a rerun skips training.
CARRIED_TRAIN_KEYS = ("sec_per_epoch", "sec_per_epoch_applio", "steps_per_epoch", "pretrained", "vram_before_mib")
# Step-record bookkeeping keys that are not results.
STEP_META_KEYS = ("status", "started_at", "finished_at", "duration_s", "error")

# Keys whose change invalidates already-produced Applio artifacts for the same run_id.
PREPARE_KEYS = ("track", "dataset_dir", "preprocess", "extract")

# rvc/infer/infer.py applies index_path.replace("trained", "added") to the whole path.
INDEX_PATH_TRAP = "trained"
# Per combo folder: clip_id -> sha256 and transpose of the source clip each output was converted from.
PROVENANCE_FILE = "_sources.json"


class RunnerError(Exception):
    """A failed step. The message is shown to the user."""


class RefusedError(RunnerError):
    """A step refused before running anything; the step record is left untouched."""


# --------------------------------------------------------------------------- config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Applio RVC experiment defined by a JSON config and record it under runs/<run_id>/."
    )
    parser.add_argument("config", type=Path, help="Experiment config, e.g. configs/baseline.json.")
    parser.add_argument(
        "steps",
        nargs="+",
        choices=[*STEPS, "all"],
        help="Steps to run in order: prepare (preprocess+extract), train (+index), infer, or all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running anything.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reuse an existing Applio logs/<model> even when its settings differ from the config.",
    )
    parser.add_argument(
        "--checkpoints",
        help="Override infer.checkpoints: 'all', 'last', or comma-separated epochs such as 10,20.",
    )
    parser.add_argument("--sets", help="Override infer.sets, comma-separated (source_eval,heldout).")
    return parser.parse_args()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise RunnerError(f"Config inheritance cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    if not path.exists():
        raise RunnerError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    base_name = data.pop("base", None)
    if base_name is None:
        return data
    base = load_config(path.parent / base_name, (*seen, path))
    return deep_merge(base, data)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(f"Invalid config: {message}")


def validate_config(cfg: dict[str, Any]) -> None:
    for key in ("run_id", "track", "dataset_dir", "preprocess", "extract", "train", "infer"):
        require(key in cfg, f"missing key '{key}'")
    require(bool(RUN_ID_PATTERN.match(cfg["run_id"])), f"run_id has unsupported characters: {cfg['run_id']!r}")
    model = cfg.get("model_run_id", cfg["run_id"])
    require(bool(RUN_ID_PATTERN.match(model)), f"model_run_id has unsupported characters: {model!r}")
    for key, value in (("run_id", cfg["run_id"]), ("model_run_id", model)):
        require(
            INDEX_PATH_TRAP not in value,
            f"{key} {value!r} contains '{INDEX_PATH_TRAP}': Applio's infer.py rewrites 'trained' to 'added' in the "
            "index path and silently runs without the index",
        )
    require(cfg["track"] in ("private", "public"), "track must be 'private' or 'public'")

    pre, ext, train, infer = cfg["preprocess"], cfg["extract"], cfg["train"], cfg["infer"]
    require(pre["sample_rate"] in (32000, 40000, 48000), "preprocess.sample_rate must be 32000/40000/48000")
    require(pre["cut_preprocess"] in ("Skip", "Simple", "Automatic"), "preprocess.cut_preprocess")
    require(pre["normalization_mode"] in ("none", "pre", "post"), "preprocess.normalization_mode")
    require(str(float(pre["chunk_len"])) in [str(i * 0.5) for i in range(1, 11)], "preprocess.chunk_len")
    require(str(float(pre["overlap_len"])) in ("0.0", "0.1", "0.2", "0.3", "0.4"), "preprocess.overlap_len")
    require(1 <= int(pre["cpu_cores"]) <= 64, "preprocess.cpu_cores must be 1..64")

    require(ext["f0_method"] in ("crepe", "rmvpe", "fcpe"), "extract.f0_method must be crepe/rmvpe/fcpe")
    require(1 <= int(ext["cpu_cores"]) <= 64, "extract.cpu_cores must be 1..64")
    require(str(ext["gpu"]) != "-", "extract.gpu '-' means CPU; pin a GPU id such as \"0\"")
    if ext["embedder_model"] == "custom":
        require(bool(ext.get("embedder_model_custom")), "extract.embedder_model_custom is required for 'custom'")

    require(train["vocoder"] in ("HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"), "train.vocoder")
    require(1 <= int(train["save_every_epoch"]) <= 100, "train.save_every_epoch must be 1..100")
    require(1 <= int(train["total_epoch"]) <= 10000, "train.total_epoch must be 1..10000")
    require(1 <= int(train["batch_size"]) <= 50, "train.batch_size must be 1..50")
    if int(train["total_epoch"]) % int(train["save_every_epoch"]):
        print(
            "WARNING: train.total_epoch is not a multiple of save_every_epoch, so the final epoch has no G_/D_ "
            "checkpoint; extending training later would resume from an earlier epoch."
        )
    require(train["index_algorithm"] in ("Auto", "Faiss", "KMeans"), "train.index_algorithm")
    pretrained = train["pretrained"]
    require(
        pretrained in ("default", "none") or (isinstance(pretrained, dict) and {"g", "d"} <= pretrained.keys()),
        "train.pretrained must be 'default', 'none' or {\"g\": path, \"d\": path}",
    )

    require(isinstance(infer["grid"], list) and infer["grid"], "infer.grid must be a non-empty list")
    for entry in infer["grid"]:
        require(
            {"f0_method", "index_rate", "protect", "volume_envelope"} <= entry.keys(),
            f"infer.grid entry needs f0_method/index_rate/protect/volume_envelope: {entry}",
        )
        require(entry["f0_method"] in ("crepe", "rmvpe", "fcpe"), "infer f0_method must be crepe/rmvpe/fcpe (no hybrid)")
        require(0 <= entry["index_rate"] <= 1, "infer index_rate must be 0..1")
        require(0 <= entry["protect"] <= 0.5, "infer protect must be 0..0.5")
        require(0 <= entry["volume_envelope"] <= 1, "infer volume_envelope must be 0..1")
    require(all(s in EVAL_SETS for s in infer["sets"]), f"infer.sets must be a subset of {EVAL_SETS}")
    parse_checkpoint_spec(infer["checkpoints"])


def parse_checkpoint_spec(spec: Any) -> str | list[int]:
    if isinstance(spec, str) and spec in ("all", "last"):
        return spec
    if isinstance(spec, str):
        try:
            return sorted({int(part) for part in spec.split(",") if part.strip()})
        except ValueError as error:
            raise RunnerError(f"Invalid checkpoint list: {spec!r}") from error
    if isinstance(spec, list) and all(isinstance(e, int) for e in spec):
        return sorted(set(spec))
    raise RunnerError(f"checkpoints must be 'all', 'last' or a list of epochs, got {spec!r}")


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.checkpoints:
        cfg["infer"]["checkpoints"] = args.checkpoints
    if args.sets:
        cfg["infer"]["sets"] = [s.strip() for s in args.sets.split(",") if s.strip()]


def combo_tag(entry: dict[str, Any]) -> str:
    return (
        f"f0-{entry['f0_method']}_ir{entry['index_rate']:.2f}"
        f"_p{entry['protect']:.2f}_ve{entry['volume_envelope']:.2f}"
    )


# --------------------------------------------------------------------------- helpers


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def dir_size_mib(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1 << 20), 1)


def dataset_files(dataset_dir: Path) -> list[Path]:
    return sorted(p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def dataset_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def applio_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in APPLIO_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    scripts_dir = str(APPLIO_PY.parent)
    if scripts_dir.lower() not in env.get("PATH", "").lower():
        env["PATH"] = f"{scripts_dir};{env.get('PATH', '')}"
    env["APPLIO_LOGS_DIR"] = str(APPLIO_LOGS)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


# --------------------------------------------------------------------------- environment


def gpu_info() -> dict[str, Any]:
    out = run_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return {"name": None, "driver": None, "memory_total_mib": None}
    name, driver, total = [part.strip() for part in out.splitlines()[0].split(",")]
    return {"name": name, "driver": driver, "memory_total_mib": int(float(total))}


def gpu_memory_used_mib() -> int | None:
    out = run_text(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"])
    try:
        return int(float(out.splitlines()[0])) if out else None
    except ValueError:
        return None


def applio_precision() -> str | None:
    value = None
    template = APPLIO_DIR / "assets" / "config_template.json"
    user = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Applio" / "config.json"
    for path in (template, user):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("precision", value)
        except (OSError, json.JSONDecodeError):
            pass
    return value


def applio_torch_version() -> str | None:
    version_file = APPLIO_DIR / "env" / "Lib" / "site-packages" / "torch" / "version.py"
    try:
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", version_file.read_text(encoding="utf-8"))
    except OSError:
        return None
    return match.group(1) if match else None


def collect_environment() -> dict[str, Any]:
    status = run_text(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    try:
        template = json.loads((APPLIO_DIR / "assets" / "config_template.json").read_text(encoding="utf-8"))
        applio_version = template.get("version")
    except (OSError, json.JSONDecodeError):
        applio_version = None
    return {
        "repo_commit": run_text(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "repo_dirty": bool(status) if status is not None else None,
        "applio_version": applio_version,
        "applio_git": run_text(["git", "describe", "--tags", "--always", "--dirty"], cwd=APPLIO_DIR),
        "applio_precision": applio_precision(),
        "gpu": gpu_info(),
        "python_runner": sys.version.split()[0],
        "python_applio": run_text([str(APPLIO_PY), "--version"]),
        "torch_applio": applio_torch_version(),
        "platform": platform.platform(),
    }


# --------------------------------------------------------------------------- run record


class RunRecord:
    """runs/<run_id>/: run.json, commands.log, logs/<step>.log, applio/."""

    def __init__(self, run_id: str, dry_run: bool) -> None:
        self.dir = RUNS_DIR / run_id
        self.path = self.dir / "run.json"
        self.dry_run = dry_run
        self.data: dict[str, Any] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.previous: dict[str, Any] = copy.deepcopy(self.data)  # state before this invocation

    def save(self) -> None:
        if self.dry_run:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = now_iso()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def log_path(self, name: str) -> Path:
        return self.dir / "logs" / f"{name}.log"

    def log_command(self, step: str, command: list[str], equivalent: list[str] | None = None) -> None:
        print(f"[{step}] $ {format_command(command)}")
        if equivalent:
            print(f"[{step}]   (core.py equivalent) {format_command(equivalent)}")
        if self.dry_run:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "commands.log").open("a", encoding="utf-8") as stream:
            stream.write(f"# {now_iso()} step={step} cwd={APPLIO_DIR}\n{format_command(command)}\n")
            if equivalent:
                stream.write(f"# core.py equivalent:\n# {format_command(equivalent)}\n")
            stream.write("\n")

    def step(self, name: str) -> dict[str, Any]:
        return self.data.setdefault("steps", {}).setdefault(name, {})

    def copy_applio_files(self, model_dir: Path) -> None:
        if self.dry_run:
            return
        target = self.dir / "applio"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "model_info.json"):
            if (model_dir / name).exists():
                shutil.copy2(model_dir / name, target / name)


@dataclass
class Context:
    cfg: dict[str, Any]
    config_path: Path
    record: RunRecord
    force: bool
    dry_run: bool
    argv: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.cfg["run_id"]

    @property
    def model(self) -> str:
        return self.cfg.get("model_run_id", self.cfg["run_id"])

    @property
    def model_dir(self) -> Path:
        return APPLIO_LOGS / self.model

    @property
    def sample_rate(self) -> int:
        return int(self.cfg["preprocess"]["sample_rate"])

    @property
    def dataset_dir(self) -> Path:
        return (REPO_ROOT / self.cfg["dataset_dir"]).resolve()


# --------------------------------------------------------------------------- subprocess tee


class LineScanner:
    """Collects lines matching failure patterns and forwards each line to optional hooks."""

    def __init__(
        self,
        patterns: list[str],
        hooks: list[Callable[[str], None]] | None = None,
        tolerated: list[str] | None = None,
    ) -> None:
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.tolerated_patterns = [re.compile(p, re.IGNORECASE) for p in tolerated or []]
        self.hooks = hooks or []
        self.matches: list[str] = []
        self.tolerated: list[str] = []

    def __call__(self, line: str) -> None:
        if any(p.search(line) for p in self.tolerated_patterns):
            self.tolerated.append(line.strip())
        elif len(self.matches) < 50 and any(p.search(line) for p in self.patterns):
            self.matches.append(line.strip())
        for hook in self.hooks:
            hook(line)


def emit_console(text: str, end: str) -> None:
    try:
        sys.stdout.write(text + end)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode("ascii", "replace").decode("ascii") + end)
    sys.stdout.flush()


def run_logged(command: list[str], log_path: Path, on_line: Callable[[str], None]) -> int:
    """Run a command in Applio/, tee its merged stdout/stderr to the console and log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {now_iso()} $ {format_command(command)}\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=APPLIO_DIR,
            env=applio_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            pump_output(proc, log, on_line)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            log.write(f"===== interrupted, exit code {proc.returncode}\n")
            raise
        returncode = proc.wait()
        log.write(f"===== exit code {returncode}\n")
    return returncode


def pump_output(proc: subprocess.Popen, log: Any, on_line: Callable[[str], None]) -> None:
    """Split on \\n and \\r; progress-bar (\\r) updates go to the console, only final lines to the log."""
    eol = re.compile(rb"\r\n|\r|\n")
    buffer = b""
    progress: str | None = None
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read1(65536)
        eof = not chunk
        buffer += chunk
        while True:
            match = eol.search(buffer)
            if not match or (match.group() == b"\r" and match.end() == len(buffer) and not eof):
                break
            text = buffer[: match.start()].decode("utf-8", "replace")
            buffer = buffer[match.end() :]
            if match.group() == b"\r":
                emit_console(text, "\r")
                progress = text
                continue
            if not text and progress:
                text = progress
            progress = None
            emit_console(text, "\n")
            log.write(text + "\n")
            log.flush()
            on_line(text)
        if eof:
            break
    tail = buffer.decode("utf-8", "replace") or (progress or "")
    if tail:
        emit_console(tail, "\n")
        log.write(tail + "\n")
        on_line(tail)


def run_applio(
    ctx: Context,
    step: str,
    log_name: str,
    command: list[str],
    equivalent: list[str] | None = None,
    hooks: list[Callable[[str], None]] | None = None,
) -> list[str]:
    """Run one Applio script and raise on failure text. Returns the tolerated lines it printed."""
    ctx.record.log_command(step, command, equivalent)
    if ctx.dry_run:
        return []
    scanner = LineScanner(FAILURE_PATTERNS[step], hooks, TOLERATED_PATTERNS.get(step))
    returncode = run_logged(command, ctx.record.log_path(log_name), scanner)
    if returncode != 0 or scanner.matches:
        details = "\n  ".join(scanner.matches[:10]) or "(no failure text matched)"
        raise RunnerError(
            f"{step} failed (exit code {returncode}). See {rel(ctx.record.log_path(log_name))}.\n  {details}"
        )
    return scanner.tolerated


def check_extract_failures(ctx: Context, tolerated: list[str]) -> list[str]:
    """Accept a small share of per-file pitch failures; return the failed slice names."""
    failed = sorted({Path(m.group(1)).stem for m in map(EXTRACT_FAILED_FILE.search, tolerated) if m})
    if not failed:
        return []
    totals = [int(m.group(2)) for m in map(EXTRACT_FAILED_LINE.search, tolerated) if m]
    total = totals[-1] if totals else 0
    if not total or len(failed) / total > EXTRACT_MAX_FAILED_FRACTION:
        raise RunnerError(
            f"extract failed for {len(failed)}/{total or '?'} slice(s) (limit {EXTRACT_MAX_FAILED_FRACTION:.0%}). "
            f"See {rel(ctx.record.log_path('extract'))}."
        )
    print(f"[extract] WARNING: pitch extraction failed for {len(failed)}/{total} slice(s); "
          "Applio leaves them out of filelist.txt", flush=True)
    return failed


# --------------------------------------------------------------------------- command builders


def applio_script(*parts: str) -> str:
    return str(APPLIO_DIR.joinpath(*parts))


def preprocess_commands(ctx: Context) -> tuple[list[str], list[str]]:
    pre = ctx.cfg["preprocess"]
    direct = [
        str(APPLIO_PY),
        applio_script("rvc", "train", "preprocess", "preprocess.py"),
        str(ctx.model_dir),
        str(ctx.dataset_dir),
        str(ctx.sample_rate),
        str(int(pre["cpu_cores"])),
        pre["cut_preprocess"],
        str(bool(pre["process_effects"])),
        str(bool(pre["noise_reduction"])),
        str(float(pre["noise_reduction_strength"])),
        str(float(pre["chunk_len"])),
        str(float(pre["overlap_len"])),
        pre["normalization_mode"],
    ]
    core = [
        "env\\python.exe", "core.py", "preprocess",
        "--model-name", ctx.model,
        "--dataset-path", str(ctx.dataset_dir),
        "--sample-rate", str(ctx.sample_rate),
        "--cpu-cores", str(int(pre["cpu_cores"])),
        "--cut-preprocess", pre["cut_preprocess"],
        "--noise-reduction-strength", str(float(pre["noise_reduction_strength"])),
        "--chunk-len", str(float(pre["chunk_len"])),
        "--overlap-len", str(float(pre["overlap_len"])),
        "--normalization-mode", pre["normalization_mode"],
    ]
    if pre["process_effects"]:
        core.append("--process-effects")
    if pre["noise_reduction"]:
        core.append("--noise-reduction")
    return direct, core


def extract_commands(ctx: Context) -> tuple[list[str], list[str]]:
    ext = ctx.cfg["extract"]
    custom = ext.get("embedder_model_custom")
    direct = [
        str(APPLIO_PY),
        applio_script("rvc", "train", "extract", "extract.py"),
        str(ctx.model_dir),
        ext["f0_method"],
        str(int(ext["cpu_cores"])),
        str(ext["gpu"]),
        str(ctx.sample_rate),
        ext["embedder_model"],
        str(custom),
        str(int(ext["include_mutes"])),
    ]
    core = [
        "env\\python.exe", "core.py", "extract",
        "--model-name", ctx.model,
        "--sample-rate", str(ctx.sample_rate),
        "--f0-method", ext["f0_method"],
        "--embedder-model", ext["embedder_model"],
        "--gpu", str(ext["gpu"]),
        "--cpu-cores", str(int(ext["cpu_cores"])),
        "--include-mutes", str(int(ext["include_mutes"])),
    ]
    if custom:
        core += ["--embedder-model-custom", str(custom)]
    return direct, core


def pretrained_paths(train: dict[str, Any], sample_rate: int) -> tuple[str, str]:
    pretrained = train["pretrained"]
    if pretrained == "none":
        return "", ""
    if isinstance(pretrained, dict):
        return str((REPO_ROOT / pretrained["g"]).resolve()), str((REPO_ROOT / pretrained["d"]).resolve())
    folder = {"HiFi-GAN": "hifi-gan", "RefineGAN": "refinegan"}.get(train["vocoder"])
    if folder is None:
        raise RunnerError(f"No default pretrain for vocoder {train['vocoder']}; set train.pretrained explicitly.")
    base = APPLIO_DIR / "rvc" / "models" / "pretraineds" / folder
    khz = sample_rate // 1000
    return str(base / f"f0G{khz}k.pth"), str(base / f"f0D{khz}k.pth")


def train_commands(ctx: Context) -> tuple[list[str], list[str]]:
    train = ctx.cfg["train"]
    pretrain_g, pretrain_d = pretrained_paths(train, ctx.sample_rate)
    direct = [
        str(APPLIO_PY),
        applio_script("rvc", "train", "train.py"),
        ctx.model,
        str(int(train["save_every_epoch"])),
        str(int(train["total_epoch"])),
        pretrain_g,
        pretrain_d,
        str(train["gpu"]),
        str(int(train["batch_size"])),
        str(ctx.sample_rate),
        "False",  # save_only_latest: keep every checkpoint for E1
        "True",  # save_every_weights: export <model>_<E>e_<S>s.pth every save_every_epoch
        str(bool(train["cache_data_in_gpu"])),
        "False",  # cleanup: never wipe a run; use a new run_id instead
        train["vocoder"],
        str(bool(train["checkpointing"])),
    ]
    core = [
        "env\\python.exe", "core.py", "train",
        "--model-name", ctx.model,
        "--vocoder", train["vocoder"],
        "--sample-rate", str(ctx.sample_rate),
        "--total-epoch", str(int(train["total_epoch"])),
        "--save-every-epoch", str(int(train["save_every_epoch"])),
        "--batch-size", str(int(train["batch_size"])),
        "--gpu", str(train["gpu"]),
        "--index-algorithm", train["index_algorithm"],
    ]
    if train["pretrained"] == "none":
        core.append("--no-pretrained")
    elif isinstance(train["pretrained"], dict):
        core += ["--custom-pretrained", "--g-pretrained-path", pretrain_g, "--d-pretrained-path", pretrain_d]
    if train["checkpointing"]:
        core.append("--checkpointing")
    if train["cache_data_in_gpu"]:
        core.append("--cache-data-in-gpu")
    return direct, core


def index_commands(ctx: Context) -> tuple[list[str], list[str]]:
    algorithm = ctx.cfg["train"]["index_algorithm"]
    direct = [
        str(APPLIO_PY),
        applio_script("rvc", "train", "process", "extract_index.py"),
        str(ctx.model_dir),
        algorithm,
    ]
    core = ["env\\python.exe", "core.py", "index", "--model-name", ctx.model, "--index-algorithm", algorithm]
    return direct, core


def index_path(ctx: Context) -> Path:
    path = ctx.model_dir / f"{ctx.model}.index"
    if INDEX_PATH_TRAP in str(path):
        raise RefusedError(
            f"Index path {path} contains '{INDEX_PATH_TRAP}'; Applio's infer.py would rewrite it to 'added' "
            "and silently convert without the index. Move the repo to a path without that word."
        )
    return path


def batch_infer_command(
    ctx: Context, input_dir: Path, output_dir: Path, pth: Path, entry: dict[str, Any], pitch: int
) -> list[str]:
    ext = ctx.cfg["extract"]
    command = [
        str(APPLIO_PY), "core.py", "batch-infer",
        "--input-folder", str(input_dir),
        "--output-folder", str(output_dir),
        "--pth-path", str(pth),
        "--index-path", str(index_path(ctx)),
        "--f0-method", entry["f0_method"],
        "--pitch", str(int(pitch)),
        "--index-rate", f"{float(entry['index_rate'])}",
        "--protect", f"{float(entry['protect'])}",
        "--volume-envelope", f"{float(entry['volume_envelope'])}",
        "--embedder-model", ext["embedder_model"],
        "--export-format", "WAV",
        "--sid", "0",
    ]
    if ext["embedder_model"] == "custom":
        command += ["--embedder-model-custom", str(ext["embedder_model_custom"])]
    return command


# --------------------------------------------------------------------------- guards


def training_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    train = {k: v for k, v in cfg["train"].items() if k != "total_epoch"}
    return {**{k: cfg[k] for k in PREPARE_KEYS}, "train": train}


def weight_files(model_dir: Path, model: str) -> dict[int, Path]:
    pattern = re.compile(rf"^{re.escape(model)}_(\d+)e_(\d+)s\.pth$")
    found: dict[int, Path] = {}
    for path in model_dir.glob(f"{model}_*e_*s.pth") if model_dir.exists() else []:
        match = pattern.match(path.name)
        if match:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


def has_resume_checkpoints(model_dir: Path) -> bool:
    return any(model_dir.glob("G_*.pth")) and any(model_dir.glob("D_*.pth"))


def refuse_or_warn(ctx: Context, problems: list[str], hint: str) -> None:
    if not problems:
        return
    text = "\n  - ".join(problems)
    if ctx.force:
        print(f"WARNING (--force): proceeding despite:\n  - {text}")
        return
    message = f"Refusing to continue:\n  - {text}\n{hint}"
    if ctx.dry_run:
        print(f"DRY-RUN would refuse: {message}")
        return
    raise RefusedError(message)


def check_config_unchanged(ctx: Context) -> None:
    """A run_id is bound to the data/feature/training settings it was first run with."""
    previous = ctx.record.previous.get("config")
    if not previous or not ctx.model_dir.exists():
        return
    old, new = training_signature(previous), training_signature(ctx.cfg)
    changed = sorted(k for k in new if old.get(k) != new.get(k))
    problems = [f"'{k}' differs from runs/{ctx.run_id}/run.json" for k in changed]
    refuse_or_warn(ctx, problems, "Use a new run_id for a new experiment (or --force to overwrite the record).")


def wanted_embedder(cfg: dict[str, Any]) -> str:
    """The embedder name Applio records in model_info.json (the custom path for 'custom')."""
    ext = cfg["extract"]
    return ext["embedder_model_custom"] if ext["embedder_model"] == "custom" else ext["embedder_model"]


def model_embedder(model_dir: Path) -> str | None:
    info_path = model_dir / "model_info.json"
    if not info_path.exists():
        return None
    return json.loads(info_path.read_text(encoding="utf-8")).get("embedder_model")


def artifact_mismatches(ctx: Context) -> list[str]:
    """Differences between Applio/logs/<model> (embedder, sample rate) and the config."""
    problems = []
    embedder, wanted = model_embedder(ctx.model_dir), wanted_embedder(ctx.cfg)
    if embedder and embedder != wanted:
        problems.append(f"model_info.json embedder_model={embedder!r}, config wants {wanted!r}")
    config_path = ctx.model_dir / "config.json"
    if config_path.exists():
        sr = json.loads(config_path.read_text(encoding="utf-8")).get("data", {}).get("sample_rate")
        if sr != ctx.sample_rate:
            problems.append(f"config.json sample_rate={sr}, config wants {ctx.sample_rate}")
    return problems


def check_prepare_target(ctx: Context) -> None:
    model_dir = ctx.model_dir
    if not model_dir.exists() or not any(model_dir.iterdir()):
        return
    problems = []
    last_prepare = ctx.record.previous.get("steps", {}).get("prepare", {})
    if not last_prepare:
        problems.append(f"Applio/logs/{ctx.model} already exists but runs/{ctx.run_id}/run.json has no prepare record")
    if has_resume_checkpoints(model_dir) or weight_files(model_dir, ctx.model):
        problems.append(f"Applio/logs/{ctx.model} already has trained checkpoints")
    problems += artifact_mismatches(ctx)
    fingerprint = last_prepare.get("dataset", {}).get("fingerprint")
    if last_prepare and not fingerprint:
        problems.append(
            "the last prepare record has no dataset fingerprint, so a change of the dataset since then "
            "cannot be ruled out"
        )
    if fingerprint and ctx.dataset_dir.exists():
        if fingerprint != dataset_fingerprint(dataset_files(ctx.dataset_dir)):
            problems.append(f"dataset content in {ctx.cfg['dataset_dir']} changed since the last prepare")
    refuse_or_warn(
        ctx,
        problems,
        "Applio skips existing features and never overwrites config.json/.index, so the result would mix "
        "old and new settings. Use a new run_id (or --force).",
    )


# --------------------------------------------------------------------------- step: prepare


def describe_dataset(ctx: Context) -> dict[str, Any]:
    if not ctx.dataset_dir.is_dir():
        raise RefusedError(f"dataset_dir does not exist: {ctx.cfg['dataset_dir']}")
    subdirs = [p.name for p in ctx.dataset_dir.iterdir() if p.is_dir()]
    if subdirs:
        print(f"WARNING: Applio treats sub-folders of the dataset as speaker ids: {subdirs}")
    files = dataset_files(ctx.dataset_dir)
    if not files:
        raise RefusedError(f"No {'/'.join(AUDIO_EXTENSIONS)} files in {ctx.cfg['dataset_dir']}")
    manifest = REPO_ROOT / "dataset" / ctx.cfg["track"] / "manifest.csv"
    return {
        "dir": rel(ctx.dataset_dir),
        "n_files": len(files),
        "total_mib": round(sum(f.stat().st_size for f in files) / (1 << 20), 1),
        "fingerprint": dataset_fingerprint(files),
        "manifest": rel(manifest) if manifest.exists() else None,
        "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
    }


def verify_prepare(ctx: Context, failed_slices: list[str]) -> dict[str, Any]:
    model_dir = ctx.model_dir
    slices = list((model_dir / "sliced_audios").glob("*.wav"))
    counts = {
        "sliced_audios": len(slices),
        "f0": len(list((model_dir / "f0").glob("*.npy"))),
        "f0_voiced": len(list((model_dir / "f0_voiced").glob("*.npy"))),
        "extracted": len(list((model_dir / "extracted").glob("*.npy"))),
    }
    problems = []
    if counts["sliced_audios"] == 0:
        problems.append("no sliced_audios produced")
    expected = {k: len(slices) - (len(failed_slices) if k.startswith("f0") else 0) for k in counts}
    problems += [f"{k}: {v} files, expected {expected[k]}" for k, v in counts.items() if v != expected[k]]
    for name in ("config.json", "model_info.json", "filelist.txt"):
        if not (model_dir / name).exists():
            problems.append(f"missing {name}")
    if failed_slices and (model_dir / "filelist.txt").exists():
        listed = (model_dir / "filelist.txt").read_text(encoding="utf-8")
        leaked = [name for name in failed_slices if f"/{name}.wav|" in listed]
        if leaked:
            problems.append(f"filelist.txt lists slices without features: {leaked[:5]}")
    if (model_dir / "config.json").exists():
        sr = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["data"]["sample_rate"]
        if sr != ctx.sample_rate:
            problems.append(f"config.json sample_rate {sr} != {ctx.sample_rate}")
    if problems:
        raise RunnerError("prepare produced incomplete artifacts:\n  - " + "\n  - ".join(problems))
    info = json.loads((model_dir / "model_info.json").read_text(encoding="utf-8"))
    return {"counts": counts, "model_info": info}


def prepare_up_to_date(ctx: Context, last: dict[str, Any], dataset: dict[str, Any]) -> bool:
    """True when the last prepare succeeded with this config and dataset and its artifacts are intact."""
    if last.get("status") != "ok" or not dataset:
        return False
    if last.get("dataset", {}).get("fingerprint") != dataset.get("fingerprint"):
        return False
    previous_cfg = ctx.record.previous.get("config") or {}
    if any(previous_cfg.get(key) != ctx.cfg.get(key) for key in PREPARE_KEYS):
        return False
    if any(not (ctx.model_dir / name).exists() for name in ("config.json", "model_info.json", "filelist.txt")):
        return False
    return not artifact_mismatches(ctx)


def step_prepare(ctx: Context) -> dict[str, Any]:
    if ctx.model != ctx.run_id:
        raise RefusedError(f"prepare is not allowed when model_run_id ({ctx.model}) != run_id ({ctx.run_id})")
    dataset = describe_dataset(ctx) if not ctx.dry_run or ctx.dataset_dir.is_dir() else {}
    last = ctx.record.previous.get("steps", {}).get("prepare", {})
    if not ctx.force and prepare_up_to_date(ctx, last, dataset):
        print("[prepare] Applio artifacts match the config and dataset of the last successful prepare; skipping.")
        return {**{k: v for k, v in last.items() if k not in STEP_META_KEYS}, "skipped": True}
    check_prepare_target(ctx)
    if not ctx.dry_run:
        # Saved before Applio runs so a failed prepare still guards the next rerun against dataset changes.
        ctx.record.step("prepare")["dataset"] = dataset
        ctx.record.save()
    direct, core = preprocess_commands(ctx)
    run_applio(ctx, "preprocess", "preprocess", direct, core)
    direct, core = extract_commands(ctx)
    tolerated = run_applio(ctx, "extract", "extract", direct, core)
    if ctx.dry_run:
        return {}
    failed_slices = check_extract_failures(ctx, tolerated)
    result = verify_prepare(ctx, failed_slices)
    ctx.record.copy_applio_files(ctx.model_dir)
    return {"dataset": dataset, **result, "extract_failed_slices": failed_slices}


# --------------------------------------------------------------------------- step: train


class VramSampler(threading.Thread):
    """Samples device-level nvidia-smi memory.used; includes other processes on the GPU."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.peak: int | None = None
        self.samples = 0

    def run(self) -> None:
        while not self.stop_event.is_set():
            used = gpu_memory_used_mib()
            if used is not None:
                self.peak = used if self.peak is None else max(self.peak, used)
                self.samples += 1
            self.stop_event.wait(VRAM_SAMPLE_SECONDS)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=VRAM_SAMPLE_SECONDS * 2)


class EpochParser:
    """Parses '<model> | epoch=N | step=S | time=HH:MM:SS | training_speed=H:MM:SS' lines."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.epochs: list[dict[str, Any]] = []
        self.last_wall = time.monotonic()

    def __call__(self, line: str) -> None:
        match = EPOCH_LINE.match(line.strip())
        if not match or match.group("model") != self.model:
            return
        now = time.monotonic()
        h, m, s = (int(x) for x in match.group("speed").split(":"))
        self.epochs.append(
            {
                "epoch": int(match.group("epoch")),
                "step": int(match.group("step")),
                "applio_s": h * 3600 + m * 60 + s,
                "wall_s": round(now - self.last_wall, 2),
                "first_in_invocation": not self.epochs,
            }
        )
        self.last_wall = now


def summarize_epochs(epochs: list[dict[str, Any]]) -> dict[str, Any]:
    steady = [e for e in epochs if not e["first_in_invocation"]] or epochs
    if not steady:
        return {"sec_per_epoch": None, "sec_per_epoch_applio": None, "steps_per_epoch": None}
    steps = sorted(epochs, key=lambda e: e["epoch"])
    step_deltas = [b["step"] - a["step"] for a, b in zip(steps, steps[1:]) if b["epoch"] == a["epoch"] + 1]
    return {
        "sec_per_epoch": round(statistics.median(e["wall_s"] for e in steady), 2),
        "sec_per_epoch_applio": statistics.median(e["applio_s"] for e in steady),
        "steps_per_epoch": statistics.median(step_deltas) if step_deltas else (steps[0]["step"] if steps[0]["epoch"] == 1 else None),
    }


def pretrained_record(ctx: Context) -> dict[str, Any]:
    g, d = pretrained_paths(ctx.cfg["train"], ctx.sample_rate)
    record: dict[str, Any] = {"spec": ctx.cfg["train"]["pretrained"]}
    for key, path in (("g", g), ("d", d)):
        if not path:
            continue
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            raise RunnerError(f"Pretrained {key.upper()} not found: {p}")
        record[key] = {"path": rel(p), "sha256": sha256_file(p)}
    return record


def resume_checkpoint_epoch(model_dir: Path, weights: dict[int, Path]) -> int | None:
    """Epoch of the latest G_<step>.pth, which is where Applio resumes (not the latest weight file)."""
    steps = [int(m.group(1)) for m in (re.match(r"^G_(\d+)\.pth$", p.name) for p in model_dir.glob("G_*.pth")) if m]
    if not steps:
        return None
    step = max(steps)
    weight_steps = {int(CHECKPOINT_STEP.search(p.name).group(1)): epoch for epoch, p in weights.items()}
    if step in weight_steps:
        return weight_steps[step]
    ratios = [s / e for s, e in weight_steps.items() if e > 0]
    return round(step / statistics.median(ratios)) if ratios else None


def check_stale_weights(ctx: Context, weights: dict[int, Path], resume_epoch: int | None) -> None:
    """Weights past the resume point belong to an abandoned trajectory; Applio never overwrites them."""
    if resume_epoch is None:
        return
    stale = [path.name for epoch, path in weights.items() if epoch > resume_epoch]
    problems = [
        f"{name} was exported after the last G_/D_ checkpoint (epoch {resume_epoch}); Applio retrains from there "
        "and keeps the old file" for name in stale
    ]
    refuse_or_warn(
        ctx,
        problems,
        f"Move those weight files out of Applio/logs/{ctx.model} before extending training "
        "(or use a total_epoch that is a multiple of save_every_epoch).",
    )


def step_train(ctx: Context) -> dict[str, Any]:
    if ctx.model != ctx.run_id:
        raise RefusedError(f"train is not allowed when model_run_id ({ctx.model}) != run_id ({ctx.run_id})")
    model_dir = ctx.model_dir
    total = int(ctx.cfg["train"]["total_epoch"])
    if not ctx.dry_run:
        for name in ("config.json", "filelist.txt", "model_info.json"):
            if not (model_dir / name).exists():
                raise RefusedError(f"Applio/logs/{ctx.model}/{name} missing; run 'prepare' first.")
    previous = ctx.record.step("train") if not ctx.dry_run else {}  # keeps ACCUMULATED_KEYS
    resumed = has_resume_checkpoints(model_dir)
    weights_before = weight_files(model_dir, ctx.model)
    resume_epoch = resume_checkpoint_epoch(model_dir, weights_before) if resumed else None
    result: dict[str, Any] = {"resumed": resumed, "resumed_from_epoch": resume_epoch}

    if total in weights_before:
        print(f"[train] {weights_before[total].name} already exists; training is complete, skipping.")
        last = ctx.record.previous.get("steps", {}).get("train", {})
        result = {"resumed": False, "resumed_from_epoch": None, **{k: last[k] for k in CARRIED_TRAIN_KEYS if k in last}}
        if previous.get("epochs"):
            result.update(summarize_epochs(previous["epochs"]))
        result["training_skipped"] = True
    else:
        if resumed:
            print(f"[train] G_/D_ checkpoints found in Applio/logs/{ctx.model}: "
                  f"Applio will resume after epoch {resume_epoch}.")
            check_stale_weights(ctx, weights_before, resume_epoch)
        pretrained = pretrained_record(ctx) if not ctx.dry_run else {}
        direct, core = train_commands(ctx)
        if ctx.dry_run:
            run_applio(ctx, "train", "train", direct, core)
        else:
            result["pretrained"] = pretrained
            result["vram_before_mib"] = gpu_memory_used_mib()
            parser = EpochParser(ctx.model)
            sampler = VramSampler()
            sampler.start()
            try:
                run_applio(ctx, "train", "train", direct, core, hooks=[parser])
            finally:
                sampler.stop()
                merged = {e["epoch"]: e for e in previous.get("epochs", [])}
                merged.update({e["epoch"]: e for e in parser.epochs})
                result["epochs"] = [merged[k] for k in sorted(merged)]
                peaks = [p for p in (previous.get("peak_vram_mib"), sampler.peak) if p is not None]
                result["peak_vram_mib"] = max(peaks) if peaks else None
                result["vram_samples"] = sampler.samples + previous.get("vram_samples", 0)
                result.update(summarize_epochs(result["epochs"]))
                previous.update(result)
                ctx.record.save()

    direct, core = index_commands(ctx)
    run_applio(ctx, "index", "train", direct, core)
    if ctx.dry_run:
        return {}

    weights = weight_files(model_dir, ctx.model)
    if total not in weights:
        raise RunnerError(f"train finished but final weights {ctx.model}_{total}e_*s.pth are missing.")
    if not (model_dir / f"{ctx.model}.index").exists():
        raise RunnerError(f"index file Applio/logs/{ctx.model}/{ctx.model}.index was not created.")
    ctx.record.copy_applio_files(model_dir)
    result["checkpoints"] = {str(epoch): path.name for epoch, path in weights.items()}
    result["disk_mib"] = dir_size_mib(model_dir)
    return result


# --------------------------------------------------------------------------- step: infer


@dataclass
class Clip:
    set_name: str
    clip_id: str
    path: Path
    transpose: int
    sha256: str = ""  # empty for the dry-run placeholder

    @property
    def stem(self) -> str:
        return f"{self.set_name}__{self.clip_id}"

    def provenance(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "transpose": self.transpose}


def load_clips(ctx: Context, set_name: str) -> list[Clip]:
    eval_dir = ctx.cfg.get("eval_dir") or f"dataset/{ctx.cfg['track']}"
    csv_path = REPO_ROOT / eval_dir / set_name / "clips.csv"
    if not csv_path.exists():
        raise RefusedError(f"Missing clip list: {rel(csv_path)}")
    clips = []
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = (REPO_ROOT / row["path"]).resolve()
            if not path.exists():
                raise RunnerError(f"Clip file missing: {row['path']} ({rel(csv_path)})")
            transpose = 0 if set_name == "heldout" else int(row.get("transpose") or 0)
            clips.append(Clip(set_name, row["clip_id"], path, transpose, sha256_file(path)))
    if not clips:
        raise RunnerError(f"No clips in {rel(csv_path)}")
    return clips


def select_checkpoints(ctx: Context) -> dict[int, Path]:
    spec = parse_checkpoint_spec(ctx.cfg["infer"]["checkpoints"])
    available = weight_files(ctx.model_dir, ctx.model)
    if not available and ctx.dry_run:
        train = ctx.cfg["train"]
        every, total = int(train["save_every_epoch"]), int(train["total_epoch"])
        expected = sorted({*range(every, total + 1, every), total})
        available = {e: ctx.model_dir / f"{ctx.model}_{e}e_<step>s.pth" for e in expected}
        print(f"[infer] DRY-RUN: no weights yet, assuming epochs {expected}")
    if not available:
        raise RefusedError(f"No {ctx.model}_<E>e_<S>s.pth weights in Applio/logs/{ctx.model}; run 'train' first.")
    if spec == "all":
        return available
    if spec == "last":
        last = max(available)
        return {last: available[last]}
    missing = [e for e in spec if e not in available]
    if missing:
        raise RefusedError(f"Requested epochs not found: {missing}. Available: {list(available)}")
    return {e: available[e] for e in spec}


def combo_dir(ctx: Context, set_name: str, epoch: int, combo: str) -> Path:
    return ctx.record.dir / "infer" / set_name / f"{epoch}e" / combo


def output_path(ctx: Context, clip: Clip, epoch: int, combo: str) -> Path:
    return combo_dir(ctx, clip.set_name, epoch, combo) / f"{clip.clip_id}.wav"


def read_provenance(folder: Path) -> dict[str, Any]:
    path = folder / PROVENANCE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record_provenance(folder: Path, clips: list[Clip]) -> None:
    data = read_provenance(folder)
    data.update({clip.clip_id: clip.provenance() for clip in clips})
    tmp = folder / f"{PROVENANCE_FILE}.tmp"
    tmp.write_text(json.dumps(dict(sorted(data.items())), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(folder / PROVENANCE_FILE)


def output_is_current(ctx: Context, clip: Clip, epoch: int, combo: str, provenance: dict[str, Any]) -> bool:
    """An existing output counts only when it was converted from this exact clip audio and transpose."""
    if not clip.sha256:
        return False
    return output_path(ctx, clip, epoch, combo).exists() and provenance.get(clip.clip_id) == clip.provenance()


def infer_group(ctx: Context, epoch: int, pth: Path, entry: dict[str, Any], pitch: int, clips: list[Clip]) -> int:
    combo = combo_tag(entry)
    work = ctx.record.dir / "infer" / "_tmp" / f"{epoch}e_{combo}_t{pitch:+d}"
    in_dir, out_dir = work / "in", work / "out"
    command = batch_infer_command(ctx, in_dir, out_dir, pth, entry, pitch)
    if ctx.dry_run:
        ctx.record.log_command("infer", command)
        return len(clips)
    shutil.rmtree(work, ignore_errors=True)
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    for clip in clips:
        shutil.copy2(clip.path, in_dir / f"{clip.stem}{clip.path.suffix.lower()}")
    run_applio(ctx, "infer", "infer", command)
    missing, moved = [], []
    for clip in clips:
        produced = out_dir / f"{clip.stem}_output.wav"
        if not produced.exists() or produced.stat().st_size == 0:
            missing.append(clip.stem)
            continue
        target = output_path(ctx, clip, epoch, combo)
        target.parent.mkdir(parents=True, exist_ok=True)
        produced.replace(target)
        moved.append(clip)
    for set_name in sorted({clip.set_name for clip in moved}):
        record_provenance(combo_dir(ctx, set_name, epoch, combo), [c for c in moved if c.set_name == set_name])
    if missing:
        raise RunnerError(f"batch-infer did not produce outputs for: {missing} (see {rel(work)})")
    shutil.rmtree(work, ignore_errors=True)
    return len(clips)


def check_infer_embedder(ctx: Context) -> str:
    """batch-infer loads the embedder named on its command line, not the one the model was trained with."""
    wanted, trained = wanted_embedder(ctx.cfg), model_embedder(ctx.model_dir)
    if trained is None:
        if not ctx.dry_run:
            print(f"[infer] WARNING: Applio/logs/{ctx.model}/model_info.json has no embedder_model; "
                  f"using {wanted!r} from the config.")
        return wanted
    problems = []
    if trained != wanted:
        problems.append(f"model {ctx.model} was trained with embedder {trained!r}, but extract.embedder_model "
                        f"gives {wanted!r} for inference")
    refuse_or_warn(ctx, problems, "Set extract.embedder_model (and embedder_model_custom) to match the model.")
    return wanted


def step_infer(ctx: Context) -> dict[str, Any]:
    infer = ctx.cfg["infer"]
    if not ctx.dry_run and not index_path(ctx).exists():
        raise RefusedError(f"Applio/logs/{ctx.model}/{ctx.model}.index missing; run 'train' first.")
    embedder = check_infer_embedder(ctx)
    clips: list[Clip] = []
    for set_name in infer["sets"]:
        try:
            clips += load_clips(ctx, set_name)
        except RunnerError as error:
            if not ctx.dry_run:
                raise
            print(f"[infer] DRY-RUN: {error}; showing a placeholder clip with transpose 0.")
            clips.append(Clip(set_name, "<clip_id>", Path(f"<{set_name}_clip>.wav"), 0))
    checkpoints = select_checkpoints(ctx)
    converted = skipped = stale = calls = 0
    started = time.monotonic()
    for epoch, pth in checkpoints.items():
        for entry in infer["grid"]:
            combo = combo_tag(entry)
            provenance = {s: read_provenance(combo_dir(ctx, s, epoch, combo)) for s in infer["sets"]}
            pending = [c for c in clips if not output_is_current(ctx, c, epoch, combo, provenance[c.set_name])]
            skipped += len(clips) - len(pending)
            outdated = [c for c in pending if c.sha256 and output_path(ctx, c, epoch, combo).exists()]
            if outdated:
                stale += len(outdated)
                print(f"[infer] epoch {epoch} {combo}: re-converting {len(outdated)} output(s) whose source clip "
                      f"or transpose changed or was never recorded: {', '.join(c.clip_id for c in outdated[:5])}"
                      f"{' ...' if len(outdated) > 5 else ''}")
            for pitch in sorted({c.transpose for c in pending}):
                group = [c for c in pending if c.transpose == pitch]
                print(f"[infer] epoch {epoch} {combo} pitch {pitch:+d}: {len(group)} clip(s)")
                converted += infer_group(ctx, epoch, pth, entry, pitch, group)
                calls += 1
    if ctx.dry_run:
        print(f"[infer] DRY-RUN: {skipped} up-to-date output(s) would be skipped, {stale} stale one(s) re-converted.")
    else:
        shutil.rmtree(ctx.record.dir / "infer" / "_tmp", ignore_errors=True)
    return {
        "sets": infer["sets"],
        "epochs": list(checkpoints),
        "combos": [combo_tag(e) for e in infer["grid"]],
        "embedder_model": embedder,
        "n_clips": {s: sum(c.set_name == s for c in clips) for s in infer["sets"]},
        "converted": converted,
        "skipped_existing": skipped,
        "reconverted_stale": stale,
        "batch_infer_calls": calls,
        "infer_wall_s": round(time.monotonic() - started, 1),
    }


# --------------------------------------------------------------------------- main


STEP_FUNCS: dict[str, Callable[[Context], dict[str, Any]]] = {
    "prepare": step_prepare,
    "train": step_train,
    "infer": step_infer,
}


def expand_steps(steps: list[str]) -> list[str]:
    wanted = set(STEPS) if "all" in steps else set(steps)
    return [s for s in STEPS if s in wanted]


def run_step(ctx: Context, name: str) -> None:
    record = ctx.record
    entry = record.step(name) if not ctx.dry_run else {}
    before = copy.deepcopy(entry)
    for key in [k for k in entry if k not in ACCUMULATED_KEYS]:
        del entry[key]
    started_at, started = now_iso(), time.monotonic()
    invocation = {
        "step": name,
        "started_at": started_at,
        "argv": ctx.argv,
        "forced": ctx.force,
        "repo_commit": record.data.get("environment", {}).get("repo_commit"),
        "repo_dirty": record.data.get("environment", {}).get("repo_dirty"),
    }
    entry.update({"status": "running", "started_at": started_at})
    record.save()
    status = "failed"
    try:
        result = STEP_FUNCS[name](ctx)
        status = "ok"
        entry.update(result)
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except RefusedError:
        status = "refused"
        raise
    except RunnerError as error:
        entry["error"] = str(error)
        raise
    finally:
        finished_at = now_iso()
        duration = round(time.monotonic() - started, 1)
        if status == "refused":
            entry.clear()
            entry.update(before)
            if not before:
                record.data.get("steps", {}).pop(name, None)
        else:
            if status == "ok":
                entry.pop("error", None)
            entry.update({"status": status, "finished_at": finished_at, "duration_s": duration})
        invocation.update({"finished_at": finished_at, "duration_s": duration, "status": status})
        record.data.setdefault("invocations", []).append(invocation)
        record.save()
    print(f"[{name}] done in {duration:.1f} s")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    try:
        cfg = load_config(args.config)
        apply_overrides(cfg, args)
        validate_config(cfg)
        record = RunRecord(cfg["run_id"], args.dry_run)
        ctx = Context(cfg, args.config.resolve(), record, args.force, args.dry_run, sys.argv[1:])
        steps = expand_steps(args.steps)
        print(f"run_id={ctx.run_id} model={ctx.model} steps={','.join(steps)}{' (dry-run)' if args.dry_run else ''}")
        if not APPLIO_PY.exists():
            raise RunnerError(f"Applio python not found: {APPLIO_PY}")
        check_config_unchanged(ctx)
        if not args.dry_run:
            record.data.setdefault("run_id", ctx.run_id)
            record.data.setdefault("created_at", now_iso())
            record.data.update(
                {
                    "config_path": rel(ctx.config_path),
                    "config": cfg,
                    "model_run_id": ctx.model,
                    "applio_logs": rel(ctx.model_dir),
                    "environment": collect_environment(),
                }
            )
            record.save()
        for step in steps:
            run_step(ctx, step)
    except RunnerError as error:
        sys.stdout.flush()
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
