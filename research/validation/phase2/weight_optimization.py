"""Phase 2 weight optimization utilities.

The routines in this module orchestrate iterative grid-search tuning of the
CBC-Ω² ``BiometricEngine`` weights.  They leverage the existing
``BiometricEngine.optimize_weights`` hook and capture per-cohort AUC and
sensitivity statistics so that downstream notebooks can track the optimisation
trajectory.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from core.biometric_engine import BiometricEngine, BiometricSample


FEATURE_COLUMNS = [
    "hrv_sdnn",
    "hrv_rmssd",
    "rhr",
    "sleep_duration",
    "sleep_efficiency",
    "waso",
    "context_switches",
]


REQUIRED_COLUMNS = [
    "timestamp",
    "user_id",
    "cohort",
    *FEATURE_COLUMNS,
    "label",
]


def _validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_phase2_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values(["user_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@dataclass(frozen=True)
class CohortMetrics:
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


def _row_to_sample(row: object) -> BiometricSample:
    if hasattr(row, "_asdict"):
        data = row._asdict()
    elif isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)
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


def _ingest(engine: BiometricEngine, rows: Iterable[object]) -> None:
    for row in rows:
        if hasattr(row, "_asdict"):
            data = row._asdict()
        elif isinstance(row, pd.Series):
            data = row.to_dict()
        else:
            data = dict(row)
        sample = _row_to_sample(data)
        engine.add_sample(str(data["user_id"]), sample)


def _cohort_metrics(y_true: np.ndarray, scores: np.ndarray, cohort: str) -> CohortMetrics:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    if len(fpr) == 0:
        return CohortMetrics(cohort=cohort, auc=float("nan"), sensitivity=float("nan"), specificity=float("nan"), threshold=float("nan"))

    j = tpr - fpr
    idx = int(np.argmax(j))
    return CohortMetrics(
        cohort=cohort,
        auc=float(auc(fpr, tpr)),
        sensitivity=float(tpr[idx]),
        specificity=float(1 - fpr[idx]),
        threshold=float(thresholds[idx]),
    )


def run_weight_grid_search(
    df: pd.DataFrame,
    label_column: str = "label",
    cohort_column: str = "cohort",
    n_splits: int = 5,
) -> Dict[str, object]:
    """Perform cross-validated weight optimisation."""

    _validate_columns(df)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[label_column].to_numpy(dtype=int)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds: List[Dict[str, object]] = []
    weight_tracker: Dict[str, List[float]] = defaultdict(list)
    aggregate_scores: List[float] = []
    aggregate_labels: List[int] = []
    cohort_scores: Dict[str, List[float]] = defaultdict(list)
    cohort_labels: Dict[str, List[int]] = defaultdict(list)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        engine = BiometricEngine()

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        engine.train_models(X_train, y_train)
        _ingest(engine, df_train.itertuples(index=False))

        val_rows = list(df_val.itertuples(index=False))
        val_samples = [_row_to_sample(row) for row in val_rows]
        user_ids = [str(row.user_id) for row in val_rows]

        best_weights, best_auc = engine.optimize_weights(
            X_val=X_val,
            y_val=y_val,
            user_ids=user_ids,
            samples=val_samples,
        )

        for key, value in best_weights.items():
            weight_tracker[key].append(value)

        optimized_scores = np.array([
            engine.compute_eoi(uid, sample).eoi for uid, sample in zip(user_ids, val_samples)
        ])

        fold_metrics = []
        for cohort, group in df_val.groupby(cohort_column):
            mask = df_val.index.isin(group.index)
            metrics = _cohort_metrics(
                y_true=y_val[mask],
                scores=optimized_scores[mask],
                cohort=str(cohort),
            )
            fold_metrics.append(metrics.as_dict())
            cohort_scores[str(cohort)].extend(optimized_scores[mask].tolist())
            cohort_labels[str(cohort)].extend(y_val[mask].tolist())

        folds.append({
            "fold": fold_idx,
            "weights": best_weights,
            "auc": float(best_auc),
            "cohorts": fold_metrics,
        })

        aggregate_scores.extend(optimized_scores.tolist())
        aggregate_labels.extend(y_val.tolist())

    overall_auc = float(roc_auc_score(aggregate_labels, aggregate_scores))
    overall_metrics = _cohort_metrics(np.array(aggregate_labels), np.array(aggregate_scores), cohort="overall").as_dict()

    cohort_summaries = {
        cohort: _cohort_metrics(np.array(labels), np.array(scores), cohort=cohort).as_dict()
        for cohort, labels in cohort_labels.items()
        for scores in [cohort_scores[cohort]]
    }

    mean_weights = {key: float(np.mean(values)) for key, values in weight_tracker.items()}

    return {
        "overall_auc": overall_auc,
        "overall_metrics": overall_metrics,
        "mean_weights": mean_weights,
        "folds": folds,
        "cohort_summaries": cohort_summaries,
    }


def export_results(results: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2 weight optimisation grid search")
    parser.add_argument("--data", type=Path, required=True, help="CSV file with biometric samples")
    parser.add_argument("--output", type=Path, required=True, help="Where to store optimisation results")
    parser.add_argument("--splits", type=int, default=5, help="Number of stratified folds")
    args = parser.parse_args()

    df = load_phase2_dataframe(args.data)
    results = run_weight_grid_search(df, n_splits=args.splits)
    export_results(results, args.output)


if __name__ == "__main__":
    _cli()
