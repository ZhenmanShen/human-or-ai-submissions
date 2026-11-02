"""Model registry: name -> "module:ClassName" in this package. Imports are lazy, so a model's
dependencies (e.g. torch) are only needed when that model is used.

To add a model: copy _template.py to <name>.py, add one line here, add configs/<name>.yaml.
"""
from __future__ import annotations

import importlib

REGISTRY = {
    "template": "_template:TemplateModel",
    "tfidf_lr": "tfidf_lr:TfidfLR",
}


def get_model_class(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(f"models.{module}"), cls)
