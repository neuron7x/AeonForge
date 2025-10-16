"""Train biometric production models from anonymized cohort data."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.biometric_engine import BaselineStats, BiometricEngine, BiometricSample


@dataclass
class CohortRecord:
    user_id: str
    timestamp: datetime
    sample: BiometricSample
    overload_event: int


DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")


def _parse_timestamp(value: str) -> datetime:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value}")


def load_cohort(path: Path) -> List[CohortRecord]:
    records: List[CohortRecord] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            if any(row.get(col, "").strip() == "" for col in (
                "user_id",
                "date",
                "hrv_sdnn",
                "hrv_rmssd",
                "rhr",
                "sleep_duration",
                "sleep_efficiency",
                "waso",
                "context_switches",
                "overload_event",
            )):
                # Skip incomplete measurements.
                continue

            ts = _parse_timestamp(row["date"].strip())
            sample = BiometricSample(
                timestamp=ts,
                hrv_sdnn=float(row["hrv_sdnn"]),
                hrv_rmssd=float(row["hrv_rmssd"]),
                rhr=float(row["rhr"]),
                sleep_duration=float(row["sleep_duration"]),
                sleep_efficiency=float(row["sleep_efficiency"]),
                waso=float(row["waso"]),
                context_switches=int(float(row["context_switches"])),
            )
            if not sample.validate():
                continue
            label = int(float(row["overload_event"]))
            records.append(CohortRecord(row["user_id"].strip(), ts, sample, label))
    records.sort(key=lambda r: (r.timestamp, r.user_id))
    return records


def _records_to_numpy(records: Sequence[CohortRecord]) -> Tuple[np.ndarray, np.ndarray, List[str], List[BiometricSample]]:
    features = np.array([
        [
            rec.sample.hrv_sdnn,
            rec.sample.hrv_rmssd,
            rec.sample.rhr,
            rec.sample.sleep_duration,
            rec.sample.sleep_efficiency,
            rec.sample.waso,
            rec.sample.context_switches,
        ]
        for rec in records
    ], dtype=float)
    labels = np.array([rec.overload_event for rec in records], dtype=int)
    user_ids = [rec.user_id for rec in records]
    samples = [rec.sample for rec in records]
    return features, labels, user_ids, samples


def train_from_cohort(
    cohort: Iterable[CohortRecord],
    version: str,
    output: Path,
    scaler_path: Path,
    logistic_path: Path,
    pca_path: Path,
) -> Tuple[dict, dict]:
    records = list(cohort)
    if len(records) < 100:
        raise ValueError("Not enough records for cohort training")

    eng = BiometricEngine()

    split_idx = int(len(records) * 0.8)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    X_train, y_train, _, _ = _records_to_numpy(train_records)
    X_val, y_val, val_user_ids, val_samples = _records_to_numpy(val_records)

    metrics = eng.train_models(X_train, y_train)

    # Build static per-user baselines from training data to support evaluation.
    history: defaultdict[str, List[BiometricSample]] = defaultdict(list)
    for rec in train_records:
        history[rec.user_id].append(rec.sample)
    for user_id, samples in history.items():
        if len(samples) >= eng.min_baseline_samples:
            eng.user_baselines[user_id] = BaselineStats.from_samples(samples)
    weights, auc = eng.optimize_weights(X_val, y_val, val_user_ids, val_samples)

    eoi_scores = np.array([eng.compute_eoi(uid, sample).eoi for uid, sample in zip(val_user_ids, val_samples)], dtype=float)
    thresholds = eng.calibrate_thresholds(eoi_scores, y_val, target_sensitivity=0.8)

    # The binary artifacts are intentionally written next to the manifest so that
    # they can be versioned externally (for example in object storage).  We avoid
    # checking them into source control because scikit-learn pickles are not
    # guaranteed to be forward compatible across releases, but we still persist
    # them here to make it easy for operators to publish the trained bundle.
    with scaler_path.open("wb") as handle:
        pickle.dump(eng.scaler, handle)
    with logistic_path.open("wb") as handle:
        pickle.dump(eng.logistic_model, handle)
    with pca_path.open("wb") as handle:
        pickle.dump(eng.pca_model, handle)

    summary = {
        "version": version,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "records": len(records),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "unique_users": len({rec.user_id for rec in records}),
        },
        "metrics": {
            **metrics,
            "validation_auc": auc,
        },
        "weights": weights,
        "thresholds": thresholds,
        "artifacts": {
            "scaler": scaler_path.name,
            "logistic": logistic_path.name,
            "pca": pca_path.name,
        },
    }

    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return weights, thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("configs/biometric/anonymized_cohort.csv"))
    parser.add_argument("--output", type=Path, default=Path("configs/biometric/weights.json"))
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()

    cohort = load_cohort(args.cohort)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scaler_path = output_path.with_name(f"scaler_{args.version}.pkl")
    logistic_path = output_path.with_name(f"logistic_{args.version}.pkl")
    pca_path = output_path.with_name(f"pca_{args.version}.pkl")

    train_from_cohort(cohort, args.version, output_path, scaler_path, logistic_path, pca_path)


if __name__ == "__main__":
    main()
