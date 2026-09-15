"""Physics-driven XGBoost transit vetting.

The training corpus is intentionally synthetic but astrophysically structured:
it encodes the observable signatures of planets, eclipsing binaries, grazing
binaries, stellar flares, and atmospheric drift.  Observed transit features
are extracted separately from Part 2 outputs and passed through the same five
column contract at inference time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger("xgb_validation_layer")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FEATURE_COLUMNS = [
    "transit_depth",
    "duration_ratio",
    "snr",
    "odd_even_diff",
    "secondary_depth",
]
MODEL_PATH = Path("xgb_transit_vetter.json")
LABEL_COLUMN = "label"
CLASS_NAMES = {0: "FALSE POSITIVE", 1: "CONFIRMED PLANET"}
_MODEL: xgb.XGBClassifier | None = None


def _bounded(values: np.ndarray, low: float = 0.0, high: float = np.inf) -> np.ndarray:
    return np.nan_to_num(values.astype(float), nan=0.0, posinf=high, neginf=low).clip(low, high)


def extract_physics_features(light_curve: pd.DataFrame, transit_result: dict[str, Any]) -> dict[str, float]:
    """Extract the model's five physical features from Part 1/Part 2 data."""
    detection = transit_result.get("detection", {})
    validation = transit_result.get("validation", {}).get("false_positive_checks", {})
    target = transit_result.get("target")
    curve = light_curve[light_curve["target"] == target] if target in light_curve.get("target", pd.Series()).values else light_curve
    flux = pd.to_numeric(curve.get("raw_flux", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    finite_flux = flux[np.isfinite(flux)]
    if len(finite_flux) > 2:
        baseline = np.nanmedian(finite_flux)
        robust_sample = finite_flux[finite_flux >= np.nanpercentile(finite_flux, 40)]
        out_of_transit_std = float(np.std(robust_sample)) if len(robust_sample) > 1 else float(np.std(finite_flux))
    else:
        out_of_transit_std = 0.0
    depth = float(detection.get("depth", 0.0) or 0.0) if detection.get("detected") else 0.0
    period = float(detection.get("period", 0.0) or 0.0)
    duration_days = float(detection.get("duration_days", 0.0) or 0.0)
    odd_even = float(validation.get("odd_even_depth_difference", 0.0) or 0.0)
    secondary = float(validation.get("secondary_depth", 0.0) or 0.0)
    return {
        "transit_depth": float(max(depth, 0.0)),
        "duration_ratio": float(duration_days / period) if period > 0 else 0.0,
        "snr": float(depth / out_of_transit_std) if out_of_transit_std > 0 else 0.0,
        "odd_even_diff": float(max(odd_even, 0.0)),
        "secondary_depth": float(max(secondary, 0.0)),
    }


def extract_features_from_results(results_path: str | Path = "TRANSIT_RESULTS.json", photometry_path: str | Path = "CLEANED_PHOTOMETRY.csv") -> pd.DataFrame:
    """Build one physics-feature row per target from the Part 2 artifacts."""
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    light_curve = pd.read_csv(photometry_path)
    rows = []
    for target, result in results.get("targets", {}).items():
        row = extract_physics_features(light_curve, {"target": target, **result})
        row["target"] = target
        rows.append(row)
    return pd.DataFrame(rows, columns=["target", *FEATURE_COLUMNS])


def build_astrophysical_corpus(samples_per_class: int = 400, random_state: int = 42) -> pd.DataFrame:
    """Create balanced subclasses with an intentionally imbalanced planet label."""
    rng = np.random.default_rng(random_state)

    def uniform(low: float, high: float) -> np.ndarray:
        return rng.uniform(low, high, samples_per_class)

    classes = [
        ("planet", 1, (uniform(0.005, 0.05), uniform(0.01, 0.08), uniform(8, 120), uniform(0, 0.004), uniform(0, 0.0015))),
        ("eclipsing_binary", 0, (uniform(0.05, 0.70), uniform(0.03, 0.30), uniform(5, 100), uniform(0.01, 0.30), uniform(0.01, 0.50))),
        ("grazing_binary", 0, (uniform(0.02, 0.25), uniform(0.05, 0.25), uniform(4, 60), uniform(0.01, 0.18), uniform(0.005, 0.25))),
        ("stellar_flare", 0, (uniform(0.005, 0.15), uniform(0.10, 0.60), uniform(2, 30), uniform(0.01, 0.20), uniform(0, 0.02))),
        ("atmospheric_drift", 0, (uniform(0.001, 0.08), uniform(0.12, 0.80), uniform(0.5, 12), uniform(0, 0.08), uniform(0, 0.01))),
    ]
    rows = []
    for class_name, label, values in classes:
        for index in range(samples_per_class):
            rows.append({
                "transit_depth": values[0][index],
                "duration_ratio": values[1][index],
                "snr": values[2][index],
                "odd_even_diff": values[3][index],
                "secondary_depth": values[4][index],
                "class_name": class_name,
                LABEL_COLUMN: label,
            })
    return pd.DataFrame(rows)


def _make_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=180,
        learning_rate=0.06,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        objective="binary:logistic",
        random_state=42,
        n_jobs=1,
    )


def train_and_evaluate_model(model_path: str | Path = MODEL_PATH, samples_per_class: int = 400) -> dict[str, Any]:
    """Train, evaluate, print metrics, and save the finalized XGBoost model."""
    corpus = build_astrophysical_corpus(samples_per_class=samples_per_class)
    X = corpus[FEATURE_COLUMNS]
    y = corpus[LABEL_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = _make_model(scale_pos_weight)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]], output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    model_path = Path(model_path)
    model.save_model(model_path)
    global _MODEL
    _MODEL = model
    metrics = {
        "feature_columns": FEATURE_COLUMNS,
        "corpus_rows": int(len(corpus)),
        "class_counts": {"false_positive": int((y == 0).sum()), "planet": int((y == 1).sum())},
        "scale_pos_weight": scale_pos_weight,
        "classification_report": report,
        "confusion_matrix_labels": ["FALSE POSITIVE", "CONFIRMED PLANET"],
        "confusion_matrix": matrix,
        "model_path": str(model_path),
    }
    print("\n=== Transit Vetter Training Metrics ===")
    print(classification_report(y_test, predictions, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]], zero_division=0))
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {matrix}")
    print(f"scale_pos_weight={scale_pos_weight:.3f}; model={model_path.resolve()}")
    return metrics


def _load_model(model_path: str | Path = MODEL_PATH) -> xgb.XGBClassifier:
    global _MODEL
    path = Path(model_path)
    if _MODEL is None or path != MODEL_PATH:
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        _MODEL = xgb.XGBClassifier()
        _MODEL.load_model(path)
    return _MODEL


def _vetting_flags(features: dict[str, float]) -> list[str]:
    flags = []
    if features["transit_depth"] > 0.05:
        flags.append("TRANSIT_TOO_DEEP_FOR_TYPICAL_PLANET")
    if features["duration_ratio"] > 0.12:
        flags.append("DURATION_RATIO_SUGGESTS_BINARY_OR_DRIFT")
    if features["snr"] < 5.0:
        flags.append("LOW_TRANSIT_SNR")
    if features["odd_even_diff"] > 0.01:
        flags.append("ODD_EVEN_DEPTH_MISMATCH")
    if features["secondary_depth"] > 0.005:
        flags.append("SECONDARY_ECLIPSE_SIGNATURE")
    return flags


def verify_transit(features_dict: dict[str, Any], model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Classify one transit candidate and return a dashboard-ready vetting result.

    ``confidence_score`` is a percentage in the range 0--100 representing the
    model probability of the ``CONFIRMED PLANET`` class.
    """
    missing = set(FEATURE_COLUMNS) - set(features_dict)
    if missing:
        raise ValueError(f"Missing transit features: {sorted(missing)}")
    features = {column: float(features_dict[column]) for column in FEATURE_COLUMNS}
    features = {column: float(_bounded(np.array([value]), 0.0)[0]) for column, value in features.items()}
    model = _load_model(model_path)
    values = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(values)[0, 1])
    flags = _vetting_flags(features)
    verdict = "CONFIRMED PLANET" if probability >= 0.5 and not ("SECONDARY_ECLIPSE_SIGNATURE" in flags and probability < 0.9) else "FALSE POSITIVE"
    return {"confidence_score": round(probability * 100.0, 4), "verdict": verdict, "vetting_flags": flags}


if __name__ == "__main__":
    train_and_evaluate_model()