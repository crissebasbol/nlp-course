"""Configuration loading, output paths and seeding."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

# Folder with configs/ (taller_8). Inside Docker the package is installed, so it is set by env.
PROJECT_DIR = Path(os.environ.get("MEDEC_PROJECT_DIR", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and apply environment overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    env_overrides = {
        ("paths", "output_dir"): "MEDEC_OUTPUT_DIR",
        ("paths", "data_csv"): "MEDEC_DATA_CSV",
        ("models", "llm_api_base"): "LLM_API_BASE",
        ("models", "llm_name"): "LLM_NAME",
    }
    for (section, key), env_name in env_overrides.items():
        if os.environ.get(env_name):
            cfg[section][key] = os.environ[env_name]
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project folder."""
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def get_output_dir(cfg: dict[str, Any]) -> Path:
    """Root folder for splits, models, predictions and reports."""
    out = resolve_path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def output_path(cfg: dict[str, Any], *parts: str) -> Path:
    """Path inside the output folder; parent folders are created."""
    path = get_output_dir(cfg).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (when installed)."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
