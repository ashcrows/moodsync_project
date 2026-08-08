"""Cross-platform helpers.

Everything OS-specific lives here so the rest of the codebase stays portable.
The `--os` CLI parameter flows into `resolve_os()`; every other module asks
this module for paths, the compute device, and worker counts.
"""
from __future__ import annotations

import multiprocessing
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

VALID_OS = ("auto", "mac", "linux", "windows")

# Map python's platform.system() output to the short names used throughout.
_SYSTEM_MAP = {
    "Darwin": "mac",
    "Linux": "linux",
    "Windows": "windows",
}


def detect_os() -> str:
    """Return the short OS name of the machine actually executing this code."""
    return _SYSTEM_MAP.get(platform.system(), "linux")


def resolve_os(requested: str | None) -> str:
    """Resolve the user-supplied --os value into a concrete OS name.

    'auto' (or None) => detect from the running machine.
    Anything else must be one of mac/linux/windows.
    """
    requested = (requested or "auto").lower()
    if requested not in VALID_OS:
        raise ValueError(
            f"--os must be one of {VALID_OS}, got {requested!r}"
        )
    if requested == "auto":
        return detect_os()
    return requested


@dataclass
class Platform:
    """Resolved, OS-aware runtime settings used across the project."""

    os_name: str            # mac | linux | windows
    running_on: str         # the actual host OS
    device: str             # cpu | cuda | mps
    num_workers: int        # safe DataLoader worker count for this OS
    def __post_init__(self):
        if self.os_name != self.running_on:
            # Warn but do not crash: the user may be generating scripts/paths
            # for a different target OS than the one they are running on.
            print(
                f"[moodsync] NOTE: --os={self.os_name} but you are running on "
                f"{self.running_on}. OS-specific paths/scripts will target "
                f"{self.os_name}; compute still runs locally on {self.running_on}.",
                file=sys.stderr,
            )

    @property
    def is_windows(self) -> bool:
        return self.os_name == "windows"

    @property
    def is_mac(self) -> bool:
        return self.os_name == "mac"


def _pick_device(prefer_gpu: bool = True) -> str:
    """Choose the best available torch device without importing torch eagerly."""
    if not prefer_gpu:
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon GPU (Metal). Guard the attribute for older torch builds.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_num_workers(os_name: str) -> int:
    """DataLoader worker count that will not deadlock on the target OS.

    Windows and macOS use 'spawn' for multiprocessing which is fragile with
    many DataLoader workers in scripts; 0 (main-process loading) is the
    reliable choice there. Linux ('fork') can use real workers.
    """
    if os_name in ("windows", "mac"):
        return 0
    cpu = multiprocessing.cpu_count()
    return max(0, min(4, cpu - 1))


def get_platform(requested_os: str | None = None, prefer_gpu: bool = True) -> Platform:
    """Build the Platform object the whole app relies on."""
    os_name = resolve_os(requested_os)
    running = detect_os()
    # On Windows, torch multiprocessing needs the freeze_support guard, so the
    # start method is set defensively when running on Windows.
    if running == "windows":
        try:
            multiprocessing.set_start_method("spawn", force=False)
        except RuntimeError:
            pass
    return Platform(
        os_name=os_name,
        running_on=running,
        device=_pick_device(prefer_gpu),
        num_workers=_safe_num_workers(os_name),
    )


def project_root() -> Path:
    """Repository root (one level above the moodsync package)."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    d = project_root() / "data_store"
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifacts_dir(tag: str | None = None) -> Path:
    """Checkpoint directory, optionally scoped to a dataset tag.

    Without a tag this is the legacy `artifacts/`. With one it is
    `artifacts/<tag>/`, which keeps the synthetic demo models from silently
    overwriting real DEAM-trained ones.
    """
    d = project_root() / "artifacts"
    if tag:
        d = d / str(tag)
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalize_path(p: str | os.PathLike) -> Path:
    """Expand ~ and env vars and return an absolute Path (portable)."""
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()
