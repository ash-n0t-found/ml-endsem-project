"""
utils/io_utils.py
=================
Logging setup, file I/O helpers, caching utilities.
Used throughout every module.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
) -> logging.Logger:
    """
    Create or retrieve a named logger with console + optional file handler.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of calling module).
    level : str
        Logging level string: DEBUG / INFO / WARNING / ERROR.
    log_file : Path, optional
        If provided, also write to this file (append mode).
    fmt : str
        Log format string.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Quick retrieval — use after setup_logger called once at entry point."""
    return logging.getLogger(name)


# ── Pickle cache ──────────────────────────────────────────────────────────────

def cache_path(cache_dir: Path, key: str) -> Path:
    """Return deterministic .pkl path for a cache key."""
    safe = hashlib.md5(key.encode()).hexdigest()[:12]
    return cache_dir / f"{safe}.pkl"


def save_cache(obj: Any, path: Path) -> None:
    """Serialize obj to pickle at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(path: Path) -> Any:
    """Deserialize pickle from path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def cached(cache_dir: Path, key: str, fn: Callable, force: bool = False) -> Any:
    """
    Load from cache if exists, else compute fn() and cache result.

    Parameters
    ----------
    cache_dir : Path
    key : str
        Human-readable cache key (will be hashed).
    fn : Callable
        Zero-argument callable that produces result.
    force : bool
        If True, recompute even if cache exists.

    Returns
    -------
    Any
    """
    p = cache_path(cache_dir, key)
    if p.exists() and not force:
        return load_cache(p)
    result = fn()
    save_cache(result, p)
    return result


# ── NumPy I/O ─────────────────────────────────────────────────────────────────

def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save named numpy arrays to compressed .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load .npz → dict of arrays."""
    data = np.load(str(path), allow_pickle=False)
    return dict(data)


# ── Timing decorator ──────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable)


def timed(logger_name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator: log wall-clock time of wrapped function.

    Usage::

        @timed("my_module")
        def expensive_fn(...): ...
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            log = logging.getLogger(logger_name or fn.__module__)
            log.info(f"{fn.__qualname__} completed in {elapsed:.2f}s")
            return result
        return wrapper  # type: ignore[return-value]
    return decorator


# ── Path helpers ──────────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if not exists. Return path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def list_files(directory: Path, pattern: str = "*") -> list[Path]:
    """Glob files in directory matching pattern."""
    return sorted(Path(directory).glob(pattern))


def subject_exists(dataset_root: Path, subject_id: str) -> bool:
    """Check BIDS subject directory exists."""
    return (dataset_root / subject_id).is_dir()


# ── Result saving ─────────────────────────────────────────────────────────────

def save_results(results: dict, path: Path) -> None:
    """Save results dict as pickle. Also logs keys saved."""
    save_cache(results, path)
    log = logging.getLogger(__name__)
    log.info(f"Results saved: {path} | keys={list(results.keys())}")


# ── Simple progress ───────────────────────────────────────────────────────────

class ProgressTracker:
    """Minimal progress tracker — no deps on tqdm."""

    def __init__(self, total: int, name: str = "Progress", logger_name: str = __name__):
        self.total = total
        self.name = name
        self.current = 0
        self._t0 = time.perf_counter()
        self.log = logging.getLogger(logger_name)

    def update(self, n: int = 1, msg: str = "") -> None:
        self.current += n
        pct = 100 * self.current / max(self.total, 1)
        elapsed = time.perf_counter() - self._t0
        eta = (elapsed / max(self.current, 1)) * (self.total - self.current)
        suffix = f" | {msg}" if msg else ""
        self.log.info(
            f"[{self.name}] {self.current}/{self.total} ({pct:.1f}%) "
            f"| elapsed={elapsed:.1f}s | ETA={eta:.1f}s{suffix}"
        )

    def done(self) -> None:
        elapsed = time.perf_counter() - self._t0
        self.log.info(f"[{self.name}] Done. Total time: {elapsed:.1f}s")

# Compatibility aliases
setup_logging = setup_logger

import json

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)