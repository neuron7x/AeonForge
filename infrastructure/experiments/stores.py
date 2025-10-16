from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


class DependencyMissingError(RuntimeError):
    """Raised when optional experiment tooling is unavailable."""


@dataclass
class RunHandle:
    backend: str
    identifier: str


class MLflowTracker:
    def __init__(self, tracking_uri: str | None = None) -> None:
        try:
            import mlflow  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
            raise DependencyMissingError("mlflow is not installed") from exc
        self._mlflow = mlflow
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def start_run(self, *, run_name: str | None = None) -> RunHandle:
        run = self._mlflow.start_run(run_name=run_name)
        return RunHandle(backend="mlflow", identifier=run.info.run_id)


class WeightsAndBiasesTracker:
    def __init__(self, project: str, entity: str | None = None, *, settings: Optional[Mapping[str, Any]] = None) -> None:
        try:
            import wandb  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
            raise DependencyMissingError("wandb is not installed") from exc
        self._wandb = wandb
        self.project = project
        self.entity = entity
        self.settings = dict(settings or {})

    def start_run(self, *, config: Optional[Mapping[str, Any]] = None) -> RunHandle:
        run = self._wandb.init(project=self.project, entity=self.entity, config=config, settings=self.settings)
        return RunHandle(backend="wandb", identifier=str(run.id))

    def log(self, metrics: Mapping[str, float]) -> None:
        self._wandb.log(metrics)


class FeastFeatureStore:
    def __init__(self, repo_path: str) -> None:
        try:
            from feast import FeatureStore  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
            raise DependencyMissingError("feast is not installed") from exc
        self._store = FeatureStore(repo_path=repo_path)

    def get_online_features(self, feature_refs: list[str], entity_rows: list[Mapping[str, Any]]) -> Dict[str, Any]:
        return self._store.get_online_features(features=feature_refs, entity_rows=entity_rows).to_dict()

    def materialize(self, start_date, end_date) -> None:
        self._store.materialize(start_date, end_date)
