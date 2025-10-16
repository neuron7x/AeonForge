"""Phase 1 validation pipeline for the CBC-Ω² biometric engine.

This module provides utilities for ingesting raw biometric CSV extracts,
training the ``BiometricEngine`` statistical models, and generating ROC
analytics per cohort.  The pipeline is intentionally lightweight so that it can
be executed either interactively inside a notebook or as a standalone script
via ``python -m research.validation.phase1.phase1_pipeline``.

Example
-------
    python -m research.validation.phase1.phase1_pipeline \
        --data data/phase1/biometrics.csv \
        --output artifacts/phase1/roc_metrics.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from core.biometric_engine import BiometricEngine, BiometricSample


BIOMETRIC_COLUMNS = [
    "timestamp",
    "user_id",
    "cohort",
    "hrv_sdnn",
    "hrv_rmssd",
    "rhr",
    "sleep_duration",
    "sleep_efficiency",
    "waso",
    "context_switches",
    "label",
]

FEATURE_COLUMNS = [
    "hrv_sdnn",
    "hrv_rmssd",
    "rhr",
    "sleep_duration",
    "sleep_efficiency",
    "waso",
    "context_switches",
]


@dataclass(frozen=True)
class CohortRocMetrics:
    """Container for ROC derived statistics."""

    cohort: str
    auc: float
    sensitivity: float
    specificity: float
    threshold: float

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "cohort": self.cohort,
            "auc": self.auc,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "threshold": self.threshold,
        }


def _validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(BIOMETRIC_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_phase1_dataframe(path: Path) -> pd.DataFrame:
    """Load and clean the Phase 1 study export.

    Parameters
    ----------
    path:
        CSV file containing one row per nightly biometric sample.

    Returns
    -------
    pd.DataFrame
        Cleaned frame with timestamps converted to pandas ``datetime64``.
    """

    df = pd.read_csv(path)
    _validate_columns(df)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values(["user_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _row_to_sample(row: object) -> BiometricSample:
    if hasattr(row, "_asdict"):
        data = row._asdict()  # pandas.NamedTuple from itertuples
    elif isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)  # fallback for mapping like objects

    return BiometricSample(
        timestamp=pd.Timestamp(data["timestamp"]).to_pydatetime(),
        hrv_sdnn=float(data["hrv_sdnn"]),
        hrv_rmssd=float(data["hrv_rmssd"]),
        rhr=float(data["rhr"]),
        sleep_duration=float(data["sleep_duration"]),
        sleep_efficiency=float(data["sleep_efficiency"]),
        waso=float(data["waso"]),
        context_switches=int(data["context_switches"]),
    )


def _ingest_baselines(engine: BiometricEngine, rows: Iterable[object]) -> None:
    for row in rows:
        if hasattr(row, "_asdict"):
            data = row._asdict()
        elif isinstance(row, pd.Series):
            data = row.to_dict()
        else:
            data = dict(row)
        sample = _row_to_sample(data)
        engine.add_sample(str(data["user_id"]), sample)


def _compute_roc_metrics(y_true: np.ndarray, scores: np.ndarray, cohort: str) -> CohortRocMetrics:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    if len(fpr) == 0:
        return CohortRocMetrics(cohort=cohort, auc=float("nan"), sensitivity=float("nan"), specificity=float("nan"), threshold=float("nan"))

    j = tpr - fpr
    idx = int(np.argmax(j))
    return CohortRocMetrics(
        cohort=cohort,
        auc=float(auc(fpr, tpr)),
        sensitivity=float(tpr[idx]),
        specificity=float(1 - fpr[idx]),
        threshold=float(thresholds[idx]),
    )


def run_cross_validated_roc(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_column: str = "label",
    cohort_column: str = "cohort",
) -> Dict[str, object]:
    """Execute stratified K-fold ROC analysis across cohorts."""

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[label_column].to_numpy(dtype=int)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds: List[Dict[str, object]] = []
    aggregate_scores: List[float] = []
    aggregate_labels: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        engine = BiometricEngine()

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        engine.train_models(X_train, y_train)
        _ingest_baselines(engine, df_train.itertuples(index=False))

        val_rows = list(df_val.itertuples(index=False))
        val_samples = [_row_to_sample(row) for row in val_rows]
        user_ids = [str(row.user_id) for row in val_rows]

        eoi_scores = np.array([
            engine.compute_eoi(uid, sample).eoi for uid, sample in zip(user_ids, val_samples)
        ])

        fold_auc = float(roc_auc_score(y_val, eoi_scores))

        cohorts = []
        for cohort, group in df_val.groupby(cohort_column):
            idxs = group.index.to_list()
            metrics = _compute_roc_metrics(
                y_true=y[idxs],
                scores=eoi_scores[np.isin(df_val.index, idxs)],
                cohort=str(cohort),
            )
            cohorts.append(metrics.as_dict())

        folds.append({
            "fold": fold_idx,
            "weights": engine.weights.copy(),
            "auc": fold_auc,
            "cohorts": cohorts,
        })

        aggregate_scores.extend(eoi_scores.tolist())
        aggregate_labels.extend(y_val.tolist())

    overall_auc = float(roc_auc_score(aggregate_labels, aggregate_scores))
    overall_metrics = _compute_roc_metrics(np.array(aggregate_labels), np.array(aggregate_scores), cohort="overall").as_dict()

    return {
        "overall_auc": overall_auc,
        "overall_metrics": overall_metrics,
        "folds": folds,
    }


def export_metrics(metrics: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 1 ROC validation pipeline")
    parser.add_argument("--data", type=Path, required=True, help="CSV file with biometric samples")
    parser.add_argument("--output", type=Path, required=True, help="Where to store ROC metrics JSON")
    parser.add_argument("--splits", type=int, default=5, help="Number of stratified folds")
    args = parser.parse_args()

    df = load_phase1_dataframe(args.data)
    metrics = run_cross_validated_roc(df, n_splits=args.splits)
    export_metrics(metrics, args.output)


if __name__ == "__main__":
    _cli()
