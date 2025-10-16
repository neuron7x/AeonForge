from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from infrastructure.experiments.stores import (
    DependencyMissingError,
    FeastFeatureStore,
    MLflowTracker,
    WeightsAndBiasesTracker,
)


def test_mlflow_tracker_import(monkeypatch):
    fake_mlflow = types.SimpleNamespace(
        set_tracking_uri=mock.Mock(),
        log_params=mock.Mock(),
        log_metrics=mock.Mock(),
        start_run=mock.Mock(return_value=types.SimpleNamespace(info=types.SimpleNamespace(run_id="123"))),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracker = MLflowTracker(tracking_uri="http://mlflow")
    tracker.log_params({"lr": 0.1})
    tracker.log_metrics({"acc": 0.9})
    run = tracker.start_run(run_name="test")
    assert run.identifier == "123"


def test_mlflow_missing_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    with pytest.raises(DependencyMissingError):
        MLflowTracker()


def test_wandb_tracker(monkeypatch):
    fake_wandb = types.SimpleNamespace(init=mock.Mock(return_value=types.SimpleNamespace(id="run")), log=mock.Mock())
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = WeightsAndBiasesTracker(project="proj")
    handle = tracker.start_run(config={"a": 1})
    tracker.log({"metric": 1.0})
    assert handle.identifier == "run"


def test_feast_feature_store(monkeypatch):
    class FakeFeatureStore:
        def __init__(self, repo_path: str) -> None:
            self.repo_path = repo_path

        def get_online_features(self, features, entity_rows):
            return types.SimpleNamespace(to_dict=lambda: {"features": features, "entities": entity_rows})

        def materialize(self, start_date, end_date):
            self.materialized = (start_date, end_date)

    monkeypatch.setitem(sys.modules, "feast", types.SimpleNamespace(FeatureStore=FakeFeatureStore))
    store = FeastFeatureStore(repo_path="/tmp")
    result = store.get_online_features(["a"], [{"id": 1}])
    assert result["features"] == ["a"]
    store.materialize("2024-01-01", "2024-01-02")
