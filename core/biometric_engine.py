from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


@dataclass
class BiometricSample:
    timestamp: datetime
    hrv_sdnn: float
    hrv_rmssd: float
    rhr: float
    sleep_duration: float
    sleep_efficiency: float
    waso: float
    context_switches: int

    def validate(self) -> bool:
        return (
            0 < self.hrv_sdnn < 200 and
            0 < self.hrv_rmssd < 150 and
            30 < self.rhr < 150 and
            0 < self.sleep_duration < 16 and
            0 <= self.sleep_efficiency <= 1 and
            0 <= self.waso < 240 and
            0 <= self.context_switches < 100
        )


@dataclass
class BaselineStats:
    sdnn_median: float
    sdnn_mad: float
    rmssd_median: float
    rmssd_mad: float
    rhr_median: float
    rhr_mad: float
    waso_median: float
    waso_mad: float

    @classmethod
    def from_samples(cls, samples: List[BiometricSample]) -> "BaselineStats":
        if len(samples) < 7:
            raise ValueError("Minimum 7 days of data required for baseline")

        sdnn = [s.hrv_sdnn for s in samples]
        rmssd = [s.hrv_rmssd for s in samples]
        rhr = [s.rhr for s in samples]
        waso = [s.waso for s in samples]

        def mad(x: List[float]) -> float:
            return float(stats.median_abs_deviation(x)) or 1.0  # avoid zero

        return cls(
            sdnn_median=float(np.median(sdnn)),
            sdnn_mad=mad(sdnn),
            rmssd_median=float(np.median(rmssd)),
            rmssd_mad=mad(rmssd),
            rhr_median=float(np.median(rhr)),
            rhr_mad=mad(rhr),
            waso_median=float(np.median(waso)),
            waso_mad=mad(waso),
        )


@dataclass
class EOIComponents:
    oi_eq: float
    oi_rr: float
    oi_pca: float
    eoi: float
    category: str

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "oi_eq": self.oi_eq,
            "oi_rr": self.oi_rr,
            "oi_pca": self.oi_pca,
            "eoi": self.eoi,
            "category": self.category,
        }


class BiometricEngine:
    """CBC-Ω² biometric processing with ROC-optimized eOI."""

    def __init__(
        self,
        baseline_window_days: int = 28,
        min_baseline_samples: int = 7,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self.baseline_window_days = baseline_window_days
        self.min_baseline_samples = min_baseline_samples

        self.user_samples: Dict[str, List[BiometricSample]] = {}
        self.user_baselines: Dict[str, BaselineStats] = {}

        self.scaler: Optional[StandardScaler] = None
        self.logistic_model: Optional[LogisticRegression] = None
        self.pca_model: Optional[PCA] = None

        self.weights = {"w_eq": 0.4, "w_rr": 0.3, "w_pca": 0.3}
        self.thresholds = {"green": 0.5, "yellow": 1.0, "orange": 1.5}

        artifact_path: Optional[Path] = None
        if config_path is not None:
            artifact_path = Path(config_path)
        else:
            # Default to repository configuration if present.
            artifact_path = Path(__file__).resolve().parents[1] / "configs" / "biometric" / "weights.json"

        if artifact_path and artifact_path.exists():
            self._load_artifacts(artifact_path)

    # ---------------------------- data management ----------------------------
    def add_sample(self, user_id: str, sample: BiometricSample) -> None:
        if not sample.validate():
            raise ValueError("Invalid biometric sample")

        arr = self.user_samples.setdefault(user_id, [])
        arr.append(sample)

        # prune by time window
        cutoff = sample.timestamp - timedelta(days=self.baseline_window_days)
        self.user_samples[user_id] = [s for s in arr if s.timestamp > cutoff]

        if len(self.user_samples[user_id]) >= self.min_baseline_samples:
            self.user_baselines[user_id] = BaselineStats.from_samples(self.user_samples[user_id])

    # ---------------------------- helpers -----------------------------------
    @staticmethod
    def _robust_z(value: float, median: float, mad: float) -> float:
        return (value - median) / (1.4826 * mad) if mad != 0 else 0.0

    @staticmethod
    def _scale_01(z: float) -> float:
        # map z in [-2,2] to [0,1]
        return float(np.clip((z + 2) / 4, 0, 1))

    def _oi_eq(self, z_sdnn: float, z_rhr: float, z_waso: float) -> float:
        return float(np.mean([self._scale_01(-z_sdnn), self._scale_01(z_rhr), self._scale_01(z_waso)]))

    def _oi_rr(self, features: np.ndarray) -> float:
        if self.scaler is None or self.logistic_model is None:
            return 0.0
        X = self.scaler.transform(features.reshape(1, -1))
        proba = float(self.logistic_model.predict_proba(X)[0][1])
        return 3.0 * proba

    def _oi_pca(self, features: np.ndarray) -> float:
        if self.scaler is None or self.pca_model is None:
            return 0.0
        X = self.scaler.transform(features.reshape(1, -1))
        pc1 = float(self.pca_model.transform(X)[0][0])
        # sigmoid to [0,3]
        return 3.0 / (1.0 + float(np.exp(-pc1)))

    # ---------------------------- public API --------------------------------
    def compute_eoi(self, user_id: str, current_sample: BiometricSample) -> EOIComponents:
        baseline = self.user_baselines.get(user_id) or BaselineStats(
            sdnn_median=50, sdnn_mad=10,
            rmssd_median=30, rmssd_mad=8,
            rhr_median=70, rhr_mad=5,
            waso_median=30, waso_mad=10,
        )

        z_sdnn = self._robust_z(current_sample.hrv_sdnn, baseline.sdnn_median, baseline.sdnn_mad)
        z_rhr  = self._robust_z(current_sample.rhr,       baseline.rhr_median,  baseline.rhr_mad)
        z_waso = self._robust_z(current_sample.waso,      baseline.waso_median, baseline.waso_mad)

        oi_eq = self._oi_eq(z_sdnn, z_rhr, z_waso)

        features = np.array([
            current_sample.hrv_sdnn,
            current_sample.hrv_rmssd,
            current_sample.rhr,
            current_sample.sleep_duration,
            current_sample.sleep_efficiency,
            current_sample.waso,
            current_sample.context_switches,
        ], dtype=float)

        oi_rr = self._oi_rr(features)
        oi_pca = self._oi_pca(features)

        eoi = (
            self.weights["w_eq"] * oi_eq +
            self.weights["w_rr"] * oi_rr +
            self.weights["w_pca"] * oi_pca
        )

        green_cutoff = float(self.thresholds.get("green", 0.0))
        yellow_cutoff = float(self.thresholds.get("yellow", green_cutoff))
        orange_cutoff = float(self.thresholds.get("orange", yellow_cutoff))

        if eoi < green_cutoff:
            category = "Green"
        elif eoi < yellow_cutoff:
            category = "Yellow"
        elif eoi < orange_cutoff:
            category = "Orange"
        else:
            category = "Red"

        return EOIComponents(oi_eq=oi_eq, oi_rr=oi_rr, oi_pca=oi_pca, eoi=float(eoi), category=category)

    # ---------------------------- ML training -------------------------------
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        self.scaler = StandardScaler().fit(X_train)
        Xs = self.scaler.transform(X_train)

        self.logistic_model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        self.logistic_model.fit(Xs, y_train)
        lr_acc = float(self.logistic_model.score(Xs, y_train))

        self.pca_model = PCA(n_components=1, random_state=42).fit(Xs)
        pca_var = float(self.pca_model.explained_variance_ratio_[0])
        return {"lr_accuracy": lr_acc, "pca_variance": pca_var}

    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        user_ids: List[str],
        samples: List[BiometricSample],
    ) -> Tuple[Dict[str, float], float]:
        best_auc = -1.0
        best = self.weights.copy()

        for w_eq in np.linspace(0, 1, 11):
            for w_rr in np.linspace(0, 1 - w_eq, int((1 - w_eq) * 10) + 1):
                w_pca = 1.0 - w_eq - w_rr
                if w_pca < 0:
                    continue
                self.weights = {"w_eq": float(w_eq), "w_rr": float(w_rr), "w_pca": float(w_pca)}
                eoi_scores = [self.compute_eoi(uid, s).eoi for uid, s in zip(user_ids, samples)]
                auc = float(roc_auc_score(y_val, eoi_scores))
                if auc > best_auc:
                    best_auc = auc
                    best = self.weights.copy()

        self.weights = best
        return best, best_auc

    def calibrate_thresholds(self, eoi_scores: np.ndarray, y_true: np.ndarray, target_sensitivity: float = 0.8) -> Dict[str, float]:
        fpr, tpr, thresholds = roc_curve(y_true, eoi_scores)
        mask = (tpr >= target_sensitivity)
        if np.any(mask):
            idx = int(np.argmax(mask))
            red = float(thresholds[idx])
        else:
            j = tpr - fpr
            idx = int(np.argmax(j))
            red = float(thresholds[idx])

        orange = red * 0.67
        yellow = red * 0.33
        self.thresholds = {"green": yellow, "yellow": orange, "orange": red, "red": red}
        return self.thresholds

    # ---------------------------- configuration -----------------------------
    def _load_artifacts(self, artifact_path: Path) -> None:
        try:
            config = json.loads(artifact_path.read_text())
        except FileNotFoundError:
            logger.warning("Biometric artifact not found at %s", artifact_path)
            return
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse biometric artifact %s: %s", artifact_path, exc)
            return

        weights = config.get("weights")
        if isinstance(weights, dict):
            self.weights.update({k: float(v) for k, v in weights.items()})

        thresholds = config.get("thresholds")
        if isinstance(thresholds, dict):
            self.thresholds.update({k: float(v) for k, v in thresholds.items()})

        artifacts = config.get("artifacts", {})
        base_dir = artifact_path.parent

        scaler_path = artifacts.get("scaler")
        logistic_path = artifacts.get("logistic")
        pca_path = artifacts.get("pca")

        if scaler_path:
            scaler = self._load_pickle(base_dir / scaler_path)
            if scaler is not None:
                self.scaler = scaler
        if logistic_path:
            logistic = self._load_pickle(base_dir / logistic_path)
            if logistic is not None:
                self.logistic_model = logistic
        if pca_path:
            pca = self._load_pickle(base_dir / pca_path)
            if pca is not None:
                self.pca_model = pca

    @staticmethod
    def _load_pickle(path: Path) -> Optional[object]:
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except FileNotFoundError:
            logger.warning("Biometric artifact component missing: %s", path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to load biometric artifact %s: %s", path, exc)
        return None
