"""Config loading: configs/default.yaml <- configs/models/<model>.yaml <- --set overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def parse_overrides(overrides: Iterable[str]) -> list[tuple[str, Any]]:
    """'model.params.C=8' -> ('model.params.C', 8). Values are parsed as YAML,
    so numbers, booleans, lists ([1,3]) and null all work."""
    parsed = []
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, raw = item.split("=", 1)
        parsed.append((key.strip(), yaml.safe_load(raw)))
    return parsed


def read_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def available_configs() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "models").glob("*.yaml"))


def load_config(model: str | None = None, config_path: str | None = None,
                overrides: Iterable[str] = ()) -> dict:
    cfg = read_yaml(CONFIG_DIR / "default.yaml")
    if config_path is None:
        if model is None:
            raise ValueError("Give either --model or --config")
        config_path = CONFIG_DIR / "models" / f"{model}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No config configs/models/{model}.yaml. Available: {', '.join(available_configs())}")
    cfg = deep_merge(cfg, read_yaml(config_path))
    for key, value in parse_overrides(overrides):
        set_nested(cfg, key, value)
    return cfg


def resolve(path: str | Path) -> Path:
    """Paths in configs are relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
