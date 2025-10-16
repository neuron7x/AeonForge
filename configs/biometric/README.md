# Biometric Engine Artifacts

The checked-in `weights.json` captures the calibrated weights and thresholds
that ship with the engine.  The supporting scaler/logistic/PCA pickles are not
committed because scikit-learn does not guarantee forward-compatible pickle
formats.  Run `python scripts/generate_biometric_weights.py` against the cohort
data to regenerate the full bundle (manifest plus pickles) when promoting new
models.
