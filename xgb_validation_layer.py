"""XGBoost validation layer for astrophysical transit candidates.

The layer separates feature construction from labels, supports a synthetic
baseline when labelled data are unavailable, evaluates precision/recall/F1,
and exposes ``validate_candidate`` for batch and interactive callers.
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
from sklearn.model_selection import GridSearchCV, train_test_split

LOGGER = logging.getLogger("xgb_validation_layer")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FEATURE_COLUMNS = [
    "transit_depth",
    "period_days",
    "duration_ratio",
    "snr",
    "odd_even_diff",
    "secondary_depth",
    "background_flux_variance",
]
LEGACY_FEATURE_COLUMNS = ["transit_depth", "duration_ratio", "snr", "odd_even_diff", "secondary_depth"]
MODEL_PATH = Path("xgb_transit_vetter.json")
LABEL_COLUMN = "label"
CLASS_NAMES = {0: "FALSE POSITIVE", 1: "CONFIRMED PLANET"}
_MODEL: xgb.XGBClassifier | None = None


def _bounded(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) and value >= 0 else np.nan


def _curve_for_target(light_curve: pd.DataFrame, target: str | None) -> pd.DataFrame:
    if target is not None and "target" in light_curve.columns:
        selected = light_curve[light_curve["target"].astype(str) == str(target)]
        if not selected.empty:
            return selected
    return light_curve


def extract_physics_features(light_curve: pd.DataFrame, transit_result: dict[str, Any]) -> dict[str, float]:
    """Extract the seven model features from a detected transit and its curve."""
    detection = transit_result.get("detection", transit_result)
    validation = transit_result.get("validation", {}).get("false_positive_checks", {})
    curve = _curve_for_target(light_curve, transit_result.get("target"))
    flux = pd.to_numeric(curve.get("raw_flux", curve.get("clean_flux", pd.Series(dtype=float))), errors="coerce").to_numpy(float)
    finite = flux[np.isfinite(flux)]
    variance = float(np.var(finite)) if len(finite) > 1 else np.nan
    
    detected = detection.get("detected", detection.get("transit_detected", False))
    depth = _bounded(detection.get("depth", np.nan)) if detected else np.nan
    period = _bounded(detection.get("period", detection.get("period_days", np.nan)))
    duration = _bounded(detection.get("duration_days", detection.get("duration", np.nan)))
    snr = _bounded(detection.get("snr", np.nan))
    
    if (np.isnan(snr) or snr == 0) and len(finite) > 2 and depth > 0:
        snr = depth / max(float(np.std(finite)), 1e-12)
        
    duration_ratio = (duration / period) if (np.isfinite(duration) and np.isfinite(period) and period > 0) else np.nan
    
    return {
        "transit_depth": depth,
        "period_days": period,
        "duration_ratio": duration_ratio,
        "snr": snr,
        "odd_even_diff": _bounded(validation.get("odd_even_depth_difference", np.nan)),
        "secondary_depth": _bounded(validation.get("secondary_depth", np.nan)),
        "background_flux_variance": variance,
    }


def extract_features_from_results(results_path: str | Path = "TRANSIT_RESULTS.json", photometry_path: str | Path = "CLEANED_PHOTOMETRY.csv") -> pd.DataFrame:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    light_curve = pd.read_csv(photometry_path)
    rows = []
    for target, result in results.get("targets", {}).items():
        row = extract_physics_features(light_curve, {"target": target, **result})
        row["target"] = target
        rows.append(row)
    return pd.DataFrame(rows, columns=["target", *FEATURE_COLUMNS])


def build_astrophysical_corpus(samples_per_class: int = 400, random_state: int = 42) -> pd.DataFrame:
    """Generate a labelled baseline corpus with physically distinct classes and realistic overlap/missingness."""
    rng = np.random.default_rng(random_state)
    rows = []
    
    for _ in range(samples_per_class):
        # 1. PLANET
        depth = np.clip(rng.lognormal(mean=np.log(0.014), sigma=0.8), 0.0005, 0.05)
        period = np.clip(rng.lognormal(mean=np.log(5.0), sigma=1.0), 0.5, 50.0)
        var = np.clip(rng.lognormal(mean=np.log(0.00002), sigma=1.0), 1e-6, 0.01)
        dur_ratio = np.clip(rng.normal(0.016, 0.01), 0.005, 0.1)
        snr = rng.uniform(15.0, 40.0)
        odd_even = np.abs(rng.normal(0, np.sqrt(var) * 2))
        sec_depth = np.abs(rng.normal(0, np.sqrt(var)))
        rows.append((depth, period, dur_ratio, snr, odd_even, sec_depth, var, "planet", 1))
        
        # 2. ECLIPSING BINARY
        depth = np.clip(rng.lognormal(mean=np.log(0.15), sigma=0.8), 0.01, 0.6)
        period = np.clip(rng.lognormal(mean=np.log(3.0), sigma=1.2), 0.2, 30.0)
        var = np.clip(rng.lognormal(mean=np.log(0.002), sigma=1.0), 1e-5, 0.05)
        dur_ratio = np.clip(rng.normal(0.1, 0.05), 0.02, 0.4)
        snr = rng.uniform(30.0, 100.0)
        odd_even = np.abs(rng.normal(0, np.sqrt(var) * 2)) if rng.random() < 0.1 else depth * rng.uniform(0.1, 0.9)
        sec_depth = depth * rng.uniform(0.05, 1.0)
        rows.append((depth, period, dur_ratio, snr, odd_even, sec_depth, var, "eclipsing_binary", 0))
        
        # 3. GRAZING BINARY
        depth = np.clip(rng.lognormal(mean=np.log(0.05), sigma=0.5), 0.005, 0.2)
        period = np.clip(rng.lognormal(mean=np.log(4.0), sigma=1.0), 0.5, 20.0)
        var = np.clip(rng.lognormal(mean=np.log(0.001), sigma=1.0), 1e-5, 0.02)
        dur_ratio = np.clip(rng.normal(0.06, 0.02), 0.02, 0.2)
        snr = rng.uniform(10.0, 60.0)
        odd_even = depth * rng.uniform(0.0, 0.5)
        sec_depth = depth * rng.uniform(0.0, 0.5)
        rows.append((depth, period, dur_ratio, snr, odd_even, sec_depth, var, "grazing_binary", 0))
        
        # 4. STELLAR FLARE / VARIABILITY
        depth = np.clip(rng.lognormal(mean=np.log(0.004), sigma=1.0), 0.0005, 0.15)
        period = np.clip(rng.lognormal(mean=np.log(4.0), sigma=1.5), 0.1, 50.0)
        var = np.clip(rng.lognormal(mean=np.log(0.00001), sigma=0.8), 1e-6, 0.1)
        dur_ratio = rng.uniform(0.005, 0.2)
        snr = rng.uniform(4.0, 7.0)
        odd_even = np.nan
        sec_depth = np.nan
        rows.append((depth, period, dur_ratio, snr, odd_even, sec_depth, var, "stellar_flare", 0))
        
        # 5. INSTRUMENTAL NOISE
        depth = np.clip(rng.lognormal(mean=np.log(0.004), sigma=1.2), 0.0005, 0.5)
        period = rng.uniform(0.1, 50.0)
        var = np.clip(rng.lognormal(mean=np.log(0.00001), sigma=1.0), 1e-6, 0.2)
        dur_ratio = rng.uniform(0.005, 0.2)
        snr = rng.uniform(4.0, 7.0)
        odd_even = np.nan
        sec_depth = np.nan
        rows.append((depth, period, dur_ratio, snr, odd_even, sec_depth, var, "instrumental_noise", 0))

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["class_name", LABEL_COLUMN])
    
    # Introduce realistic missing data for secondary validation features
    mask_odd_even = rng.random(len(df)) < 0.20
    mask_secondary = rng.random(len(df)) < 0.20
    df.loc[mask_odd_even, "odd_even_diff"] = np.nan
    df.loc[mask_secondary, "secondary_depth"] = np.nan
    
    return df


def _make_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )


def train_and_evaluate_model(model_path: str | Path = MODEL_PATH, samples_per_class: int = 400) -> dict[str, Any]:
    """Tune, evaluate, log, and save the classifier without label leakage."""
    corpus = build_astrophysical_corpus(samples_per_class=samples_per_class)
    X = corpus.loc[:, FEATURE_COLUMNS].copy()
    y = corpus.loc[:, LABEL_COLUMN].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    search = GridSearchCV(
        estimator=_make_model(scale_pos_weight),
        param_grid={"n_estimators": [120, 180], "max_depth": [2, 3], "learning_rate": [.04, .08]},
        scoring="f1",
        cv=3,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]], output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    model_path = Path(model_path)
    model.save_model(model_path)
    global _MODEL
    _MODEL = model
    LOGGER.info("XGBoost validation metrics: precision=%.4f recall=%.4f f1=%.4f", report[CLASS_NAMES[1]]["precision"], report[CLASS_NAMES[1]]["recall"], report[CLASS_NAMES[1]]["f1-score"])
    LOGGER.info("Confusion matrix [[TN, FP], [FN, TP]]: %s", matrix)
    return {
        "feature_columns": FEATURE_COLUMNS,
        "corpus_rows": int(len(corpus)),
        "class_counts": {"false_positive": int((y == 0).sum()), "planet": int((y == 1).sum())},
        "scale_pos_weight": scale_pos_weight,
        "best_params": search.best_params_,
        "classification_report": report,
        "confusion_matrix_labels": [CLASS_NAMES[0], CLASS_NAMES[1]],
        "confusion_matrix": matrix,
        "model_path": str(model_path),
    }


def _load_model(model_path: str | Path = MODEL_PATH) -> xgb.XGBClassifier:
    global _MODEL
    path = Path(model_path)
    if _MODEL is None or path.resolve() != MODEL_PATH.resolve():
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        _MODEL = xgb.XGBClassifier()
        _MODEL.load_model(path)
    return _MODEL


def _flags(features: dict[str, float]) -> list[str]:
    flags: list[str] = []
    if features["transit_depth"] > .05:
        flags.append("TRANSIT_TOO_DEEP_FOR_TYPICAL_PLANET")
    if features["period_days"] <= 0:
        flags.append("INVALID_OR_MISSING_PERIOD")
    if features["duration_ratio"] > .12:
        flags.append("DURATION_RATIO_SUGGESTS_BINARY_OR_DRIFT")
    if features["snr"] < 5:
        flags.append("LOW_TRANSIT_SNR")
    if features["odd_even_diff"] > .01:
        flags.append("ODD_EVEN_DEPTH_MISMATCH")
    if features["secondary_depth"] > .005:
        flags.append("SECONDARY_ECLIPSE_SIGNATURE")
    if features["background_flux_variance"] > .05:
        flags.append("HIGH_BACKGROUND_FLUX_VARIANCE")
    return flags


def validate_candidate(features_dict: dict[str, Any], model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Classify one candidate and return the public ML validation contract."""
    missing = [column for column in FEATURE_COLUMNS if column not in features_dict]
    if missing:
        raise ValueError(f"Missing candidate features: {missing}")
        
    features = {column: _bounded(features_dict[column]) for column in FEATURE_COLUMNS}
    
    # Early rejection: if core features are NaN or 0, upstream detection failed.
    # Do NOT call XGBoost on failed detections to avoid assigning false confidence.
    depth = features.get("transit_depth", np.nan)
    snr = features.get("snr", np.nan)
    period = features.get("period_days", np.nan)
    
    if np.isnan(depth) or np.isnan(snr) or np.isnan(period):
        return {"candidate_confidence": np.nan, "is_exoplanet_candidate": False, "flags": ["INSUFFICIENT_DATA"]}
        
    if depth == 0 or snr == 0 or period == 0:
        return {"candidate_confidence": np.nan, "is_exoplanet_candidate": False, "flags": ["VETTER_NOT_RUN"]}
        
    model = _load_model(model_path)
    values = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(values)[0, 1])
    flags = _flags(features)
    severe = {"SECONDARY_ECLIPSE_SIGNATURE", "ODD_EVEN_DEPTH_MISMATCH", "HIGH_BACKGROUND_FLUX_VARIANCE"}
    is_candidate = bool(probability >= .5 and not (severe.intersection(flags) and probability < .9))
    return {"candidate_confidence": round(probability * 100.0, 4), "is_exoplanet_candidate": is_candidate, "flags": flags}


def verify_transit(features_dict: dict[str, Any], model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Backward-compatible adapter used by the existing simulator."""
    enriched = dict(features_dict)
    for col in FEATURE_COLUMNS:
        enriched.setdefault(col, np.nan)
    result = validate_candidate(enriched, model_path=model_path)
    return {"confidence_score": result["candidate_confidence"], "verdict": "CONFIRMED PLANET" if result["is_exoplanet_candidate"] else "FALSE POSITIVE", "vetting_flags": result["flags"]}


if __name__ == "__main__":
    metrics = train_and_evaluate_model()
    print(json.dumps(metrics, indent=2, default=str))
