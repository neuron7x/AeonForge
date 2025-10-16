"""Configuration helpers for loading Mycelial modules from YAML manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from .mycelial_modules import AnalyticWorkerModule, DelegationCriticModule, LLMAgentModule


MODULE_TYPES = {
    "llm": LLMAgentModule,
    "analytics": AnalyticWorkerModule,
    "critic": DelegationCriticModule,
}


def _ensure_mapping(data: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"YAML document at {path} must define a mapping")
    return data


def _normalise_vector(values: Iterable[Any]) -> Iterable[float]:
    for value in values:
        yield float(value)


def load_mycelial_modules(config_dir: Path) -> Dict[str, object]:
    """Load module manifests from ``config_dir`` and instantiate classes."""

    modules: Dict[str, object] = {}
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_path.read_text()) or {}
        doc = _ensure_mapping(data, yaml_path)
        raw_modules = doc.get("modules", {})
        if not isinstance(raw_modules, Mapping):
            raise TypeError(f"'modules' key in {yaml_path} must be a mapping")
        for name, spec in raw_modules.items():
            if not isinstance(spec, Mapping):
                raise TypeError(f"Module specification for '{name}' in {yaml_path} must be a mapping")
            module_type = spec.get("type")
            if module_type not in MODULE_TYPES:
                raise ValueError(f"Unsupported module type '{module_type}' in {yaml_path}")
            factory = MODULE_TYPES[module_type]
            kwargs = dict(spec)
            kwargs.pop("type")
            if module_type == "llm":
                if "responses" in kwargs:
                    kwargs["responses"] = {
                        str(key): list(_normalise_vector(value)) for key, value in kwargs["responses"].items()
                    }
                if "default" in kwargs:
                    kwargs["default"] = list(_normalise_vector(kwargs["default"]))
            elif module_type == "analytics":
                if "weights" in kwargs:
                    kwargs["weights"] = list(_normalise_vector(kwargs["weights"]))
            elif module_type == "critic":
                pass
            modules[name] = factory(**kwargs)
    return modules


__all__ = ["load_mycelial_modules"]

