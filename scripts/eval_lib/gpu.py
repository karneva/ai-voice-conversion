"""Device selection, model release, and per-stage timing / VRAM logging."""

from __future__ import annotations

import gc
import shutil
import subprocess
import threading
import time
from typing import Any

import torch

PROGRESS_EVERY = 50


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def nvidia_smi_used_mib() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class _SmiPoller(threading.Thread):
    def __init__(self, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak: int | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            used = nvidia_smi_used_mib()
            if used is not None:
                self.peak = used if self.peak is None else max(self.peak, used)
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5)


class StageMonitor:
    """Times one metric stage and records peak VRAM.

    torch peak covers PyTorch models only; the nvidia-smi delta also covers
    CTranslate2 (faster-whisper) but includes any other process on the GPU.
    """

    def __init__(self, name: str, device: str) -> None:
        self.name = name
        self.cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.items = 0
        self.stats: dict[str, Any] = {}
        self._poller: _SmiPoller | None = None
        self._base_smi: int | None = None
        self._t0 = 0.0

    def tick(self, n: int = 1) -> None:
        before = self.items
        self.items += n
        if self.items // PROGRESS_EVERY > before // PROGRESS_EVERY:
            elapsed = time.perf_counter() - self._t0
            print(f"[{self.name}] {self.items} item(s), {elapsed:.0f} s elapsed", flush=True)

    def __enter__(self) -> "StageMonitor":
        release()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
            self._base_smi = nvidia_smi_used_mib()
            if self._base_smi is not None:
                self._poller = _SmiPoller()
                self._poller.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed = time.perf_counter() - self._t0
        torch_peak = torch.cuda.max_memory_allocated() / 2**20 if self.cuda else None
        smi_delta = None
        if self._poller is not None:
            self._poller.stop()
            if self._poller.peak is not None and self._base_smi is not None:
                smi_delta = self._poller.peak - self._base_smi
        release()
        self.stats = {
            "stage": self.name,
            "items": self.items,
            "seconds": round(elapsed, 2),
            "seconds_per_item": round(elapsed / self.items, 3) if self.items else None,
            "torch_peak_mib": round(torch_peak) if torch_peak is not None else None,
            "smi_peak_delta_mib": smi_delta,
        }
        per_item = f"{elapsed / self.items:.3f} s/item" if self.items else "no items"
        vram = ""
        if torch_peak is not None:
            vram = f", torch peak {torch_peak:.0f} MiB"
        if smi_delta is not None:
            vram += f", nvidia-smi peak +{smi_delta} MiB"
        print(f"[{self.name}] {self.items} item(s) in {elapsed:.1f} s ({per_item}){vram}", flush=True)
