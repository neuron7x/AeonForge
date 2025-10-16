import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from core.biometric_engine import BiometricEngine, BiometricSample
from scripts.generate_biometric_weights import load_cohort, train_from_cohort

def make_sample(ts, sdnn=50, rmssd=30, rhr=70, sd=7.0, se=0.85, waso=30, cs=5):
    return BiometricSample(
        timestamp=ts, hrv_sdnn=sdnn, hrv_rmssd=rmssd, rhr=rhr,
        sleep_duration=sd, sleep_efficiency=se, waso=waso, context_switches=cs
    )

def test_baseline_time_window():
    eng = BiometricEngine(baseline_window_days=28, min_baseline_samples=7)
    now = datetime.now()
    # 30 days of data; older than 28 should be pruned
    for i in range(30):
        s = make_sample(now - timedelta(days=30-i))
        eng.add_sample("u", s)
    assert all(
        s.timestamp >= now - timedelta(days=eng.baseline_window_days + 1)
        for s in eng.user_samples["u"]
    )
    assert "u" in eng.user_baselines

def test_eoi_components_and_category():
    eng = BiometricEngine()
    now = datetime.now()
    # baseline
    for i in range(10):
        s = make_sample(now - timedelta(days=10-i), sdnn=55+np.random.randn()*2, rmssd=32, rhr=68, waso=28)
        eng.add_sample("u", s)
    # stressed sample: lower HRV, higher RHR & WASO
    cur = make_sample(now, sdnn=40, rmssd=25, rhr=82, waso=60, cs=12)
    comp = eng.compute_eoi("u", cur)
    assert 0 <= comp.oi_eq <= 1
    assert 0 <= comp.eoi <= 3
    assert comp.category in {"Green","Yellow","Orange","Red"}

def test_train_optimize_calibrate():
    eng = BiometricEngine()
    # synth train
    X, y, users, samples = [], [], [], []
    now = datetime.now()
    for i in range(100):
        sdnn = 55 + np.random.randn()*4
        rmssd = 30 + np.random.randn()*3
        rhr = 70 + np.random.randn()*4
        waso = 30 + np.random.randn()*8
        sd = 7 + np.random.randn()*0.3
        se = 0.85 + np.random.randn()*0.03
        cs = max(0, int(5 + np.random.randn()*2))
        s = make_sample(now, sdnn=sdnn, rmssd=rmssd, rhr=rhr, sd=sd, se=se, waso=waso, cs=cs)
        X.append([sdnn,rmssd,rhr,sd,se,waso,cs])
        label = 1 if (sdnn<50 and rhr>72) or waso>40 else 0
        y.append(label)
        users.append("u")
        samples.append(s)
    X = np.array(X, dtype=float); y = np.array(y, dtype=int)
    eng.train_models(X, y)
    w, auc = eng.optimize_weights(X, y, users, samples)
    assert 0.5 < auc <= 1.0
    eois = np.array([eng.compute_eoi(u, s).eoi for u, s in zip(users, samples)], dtype=float)
    thr = eng.calibrate_thresholds(eois, y, target_sensitivity=0.8)
    assert 0 < thr["green"] < thr["yellow"] < thr["orange"]


@pytest.fixture
def drift_samples():
    now = datetime.now()
    samples = []
    for i in range(27):
        # gradual drift: HRV trending down, RHR up
        samples.append(
            make_sample(
                now - timedelta(days=27 - i),
                sdnn=60 - i * 0.4,
                rmssd=32 - i * 0.2,
                rhr=66 + i * 0.3,
                waso=28 + i * 0.4,
            )
        )
    samples.append(make_sample(now - timedelta(days=1), sdnn=51, rmssd=29, rhr=71, waso=36))
    samples.append(make_sample(now, sdnn=50, rmssd=28, rhr=71, waso=38))
    return samples


@pytest.fixture(scope="module")
def anonymized_cohort_records():
    return load_cohort(Path("configs/biometric/anonymized_cohort.csv"))


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory, anonymized_cohort_records):
    base_dir = tmp_path_factory.mktemp("biometric-artifacts")
    weights_path = base_dir / "weights.json"
    scaler_path = base_dir / "scaler_test.pkl"
    logistic_path = base_dir / "logistic_test.pkl"
    pca_path = base_dir / "pca_test.pkl"

    weights, thresholds = train_from_cohort(
        anonymized_cohort_records,
        version="unit-test",
        output=weights_path,
        scaler_path=scaler_path,
        logistic_path=logistic_path,
        pca_path=pca_path,
    )

    return {
        "weights": weights,
        "thresholds": thresholds,
        "weights_path": weights_path,
        "scaler_path": scaler_path,
        "logistic_path": logistic_path,
        "pca_path": pca_path,
    }


@pytest.fixture
def missing_data_csv(tmp_path):
    csv_path = tmp_path / "cohort.csv"
    csv_path.write_text(
        "user_id,date,hrv_sdnn,hrv_rmssd,rhr,sleep_duration,sleep_efficiency,waso,context_switches,overload_event\n"
        "U1,2024-01-01,55,32,68,7.2,0.9,28,6,0\n"
        "U2,2024-01-02, ,30,72,6.8,0.82,48,10,1\n"
    )
    return csv_path


def test_sensor_drift_baseline_adapts(drift_samples):
    eng = BiometricEngine()
    for sample in drift_samples[:-1]:
        eng.add_sample("u", sample)

    final = drift_samples[-1]
    components = eng.compute_eoi("u", final)
    acute = make_sample(
        final.timestamp + timedelta(hours=1),
        sdnn=42,
        rmssd=24,
        rhr=80,
        waso=55,
        cs=12,
    )
    acute_components = eng.compute_eoi("u", acute)

    assert components.eoi < acute_components.eoi
    assert components.eoi <= eng.thresholds["orange"] * 1.2


def test_load_cohort_skips_missing(missing_data_csv):
    cohort = load_cohort(missing_data_csv)
    assert len(cohort) == 1
    assert cohort[0].user_id == "U1"


def test_training_pipeline_produces_artifacts(trained_artifacts):
    weights_path = trained_artifacts["weights_path"]
    assert weights_path.exists()
    payload = json.loads(weights_path.read_text())

    assert payload["weights"] == pytest.approx(trained_artifacts["weights"])
    assert payload["thresholds"] == pytest.approx(trained_artifacts["thresholds"])
    assert payload["thresholds"]["red"] == pytest.approx(trained_artifacts["thresholds"]["red"])
    assert payload["artifacts"]["scaler"] == trained_artifacts["scaler_path"].name
    assert payload["artifacts"]["logistic"] == trained_artifacts["logistic_path"].name
    assert payload["artifacts"]["pca"] == trained_artifacts["pca_path"].name

    for key in ("scaler_path", "logistic_path", "pca_path"):
        assert trained_artifacts[key].exists()


def test_engine_bootstraps_from_artifacts(trained_artifacts):
    eng = BiometricEngine(config_path=trained_artifacts["weights_path"])

    assert eng.scaler is not None
    assert eng.logistic_model is not None
    assert eng.pca_model is not None
    assert eng.weights == pytest.approx(trained_artifacts["weights"])
    assert eng.thresholds["red"] == pytest.approx(trained_artifacts["thresholds"]["red"])

    now = datetime.now()
    sample = make_sample(now, sdnn=45, rmssd=28, rhr=78, waso=42, cs=10)
    components = eng.compute_eoi("unit", sample)

    assert 0 <= components.oi_rr <= 3
    assert 0 <= components.oi_pca <= 3
