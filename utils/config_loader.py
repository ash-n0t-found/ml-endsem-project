"""
config_loader.py
================
Load and validate config.yaml. Provides singleton Config object used
throughout the codebase. All path resolution handled here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ── Singleton cache ──────────────────────────────────────────────────────────
_CONFIG: Optional["Config"] = None
_CONFIG_PATH: Optional[Path] = None


class Config:
    """
    Thin wrapper around a nested dict loaded from config.yaml.
    Supports attribute-style access (cfg.eeg.filter.l_freq) via
    recursive _Section objects, plus dict-style access.
    """

    def __init__(self, data: Dict[str, Any], root_file: Path):
        self._data = data
        self._root_file = root_file
        self._resolve_paths()

    # ── Path resolution ──────────────────────────────────────────────────────
    def _resolve_paths(self) -> None:
        """Make relative output paths absolute relative to config file dir."""
        base = self._root_file.parent.parent  # repo root
        paths = self._data.get("paths", {})
        for key, val in paths.items():
            if key != "dataset_root":  # dataset path already absolute
                paths[key] = str((base / val).resolve())
        self._data["paths"] = paths

    # ── Attribute access ──────────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            val = self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no section '{name}'")
        if isinstance(val, dict):
            return _Section(val)
        return val

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def raw(self) -> Dict[str, Any]:
        """Return raw nested dict."""
        return self._data

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def dataset_root(self) -> Path:
        return Path(self._data["paths"]["dataset_root"])

    @property
    def output_root(self) -> Path:
        p = Path(self._data["paths"]["output_root"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def figures_dir(self) -> Path:
        p = Path(self._data["paths"]["figures_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_dir(self) -> Path:
        p = Path(self._data["paths"]["results_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_dir(self) -> Path:
        p = Path(self._data["paths"]["models_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_dir(self) -> Path:
        p = Path(self._data["paths"]["cache_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_dir(self) -> Path:
        p = Path(self._data["paths"]["logs_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Subject helpers ───────────────────────────────────────────────────────
    @property
    def dev_subjects(self) -> List[str]:
        return self._data["subjects"]["development"]

    @property
    def all_subjects(self) -> List[str]:
        return self._data["subjects"]["all_clean"]

    def subject_dir(self, subject_id: str) -> Path:
        """Return BIDS subject directory path."""
        return self.dataset_root / subject_id

    # ── Condition helpers ─────────────────────────────────────────────────────
    @property
    def conditions(self) -> Dict[str, Any]:
        return self._data["paradigm"]["conditions"]

    @property
    def condition_labels(self) -> Dict[str, int]:
        return {k: v["label"] for k, v in self.conditions.items()}

    @property
    def n_digits_map(self) -> Dict[str, int]:
        return {k: v["n_digits"] for k, v in self.conditions.items()}

    # ── Modality helpers ──────────────────────────────────────────────────────
    @property
    def modality_names(self) -> List[str]:
        return [m["name"] for m in self._data["modalities"]]

    @property
    def modality_pairs(self) -> List[List[str]]:
        return self._data["modality_pairs"]

    # ── Repr ──────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"Config(project='{self._data['project']['name']}', "
            f"n_dev_subjects={len(self.dev_subjects)}, "
            f"dataset_root='{self.dataset_root}')"
        )


class _Section:
    """Recursive wrapper for nested dict sections."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            val = self._data[name]
        except KeyError:
            raise AttributeError(f"Config section has no key '{name}'")
        if isinstance(val, dict):
            return _Section(val)
        return val

    def __getitem__(self, key: str) -> Any:
        val = self._data[key]
        if isinstance(val, dict):
            return _Section(val)
        return val

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        if isinstance(val, dict):
            return _Section(val)
        return val

    def raw(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"Section({list(self._data.keys())})"

    def __contains__(self, key: str) -> bool:
        return key in self._data


# ── Public API ────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load config from YAML file. Caches singleton — subsequent calls
    with same path return cached instance.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config.yaml. Defaults to <repo_root>/config/config.yaml.

    Returns
    -------
    Config
    """
    global _CONFIG, _CONFIG_PATH

    if config_path is None:
        # Default: find config.yaml relative to this file
        this_dir = Path(__file__).resolve().parent
        config_path = this_dir.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path).resolve()

    # Return cached if same path
    if _CONFIG is not None and _CONFIG_PATH == config_path:
        return _CONFIG

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _CONFIG = Config(raw, config_path)
    _CONFIG_PATH = config_path
    return _CONFIG


def reset_config() -> None:
    """Clear singleton cache (useful for testing)."""
    global _CONFIG, _CONFIG_PATH
    _CONFIG = None
    _CONFIG_PATH = None


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
    print(f"Dev subjects ({len(cfg.dev_subjects)}): {cfg.dev_subjects}")
    print(f"EEG theta ROI: {cfg.eeg.theta_roi}")
    print(f"GGM stability pi: {cfg.ggm.stability_selection.pi_threshold}")
    print(f"Modality pairs: {cfg.modality_pairs}")