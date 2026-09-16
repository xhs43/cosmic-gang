"""
evaluate_model_performance.py
------------------------------
Benchmark evaluation script for xgb_transit_vetter.json.

Steps:
1. Generate a balanced synthetic test set using build_astrophysical_corpus()
   (from xgb_validation_layer) covering 4 classes:
   Planet, Eclipsing Binary, Stellar Flare, Instrumental Noise.
2. Load xgb_transit_vetter.json and run inference.
3. Apply the astrophysical safety-flag layer.
4. Compute Precision, Recall, F1 per class + PR-AUC + FAP + Confusion Matrix.
5. Write model_evaluation_metrics.txt.
6. Print one-line weighted-F1 summary.

Does NOT retrain the model. Does NOT touch app.py or xgb_validation_layer.py.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
)

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
MODEL_PATH = ROOT / "xgb_transit_vetter.json"

# Import from the existing (unmodified) validation layer
sys.path.insert(0, str(ROOT))
from xgb_validation_layer import (
    FEATURE_COLUMNS,
    build_astrophysical_corpus,
    _flags as compute_flags,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Build balanced benchmark evaluation set
# ──────────────────────────────────────────────────────────────────────────────

SAMPLES_PER_CLASS = 200  # lightweight but statistically meaningful
EVAL_CLASSES = ["planet", "eclipsing_binary", "stellar_flare", "instrumental_noise"]

def build_eval_set(seed: int = 42) -> pd.DataFrame:
    """Return a balanced DataFrame with 4 classes, using the project corpus generator."""
    corpus = build_astrophysical_corpus(samples_per_class=SAMPLES_PER_CLASS, random_state=seed)
    # Keep only the 4 required classes
    df = corpus[corpus["class_name"].isin(EVAL_CLASSES)].copy()
    df = df.reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Load saved model and run inference
# ──────────────────────────────────────────────────────────────────────────────

def load_model() -> xgb.XGBClassifier:
    if not MODEL_PATH.is_file():
        print(f"ERROR: model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    m = xgb.XGBClassifier()
    m.load_model(MODEL_PATH)
    return m


def infer(model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary_preds, planet_proba)."""
    X_filled = X[FEATURE_COLUMNS].fillna(0.0)
    proba      = model.predict_proba(X_filled)[:, 1]
    preds      = (proba >= 0.5).astype(int)
    return preds, proba


# ──────────────────────────────────────────────────────────────────────────────
# 3. Apply astrophysical safety flags
# ──────────────────────────────────────────────────────────────────────────────

def apply_safety_layer(df: pd.DataFrame, preds: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Override predictions where severe safety flags are triggered and confidence < 90%."""
    SEVERE_FLAGS = {
        "SECONDARY_ECLIPSE_SIGNATURE",
        "ODD_EVEN_DEPTH_MISMATCH",
        "HIGH_BACKGROUND_FLUX_VARIANCE",
        "TRANSIT_TOO_DEEP_FOR_TYPICAL_PLANET",
    }
    final_preds = preds.copy()
    flag_counts: dict[str, int] = {}
    for i, row in enumerate(df[FEATURE_COLUMNS].to_dict("records")):
        feats = {k: (float(v) if pd.notna(v) else 0.0) for k, v in row.items()}
        flags = compute_flags(feats)
        for f in flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1
        severe_hit = bool(SEVERE_FLAGS.intersection(set(flags)))
        if severe_hit and proba[i] < 0.9:
            final_preds[i] = 0  # force FALSE POSITIVE
    return final_preds, flag_counts


# ──────────────────────────────────────────────────────────────────────────────
# 4. Compute metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    # Per-class report (binary: planet=1, false-positive=0)
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["FALSE POSITIVE", "CONFIRMED PLANET"],
        output_dict=True,
        zero_division=0,
    )

    # PR-AUC (for planet class)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, proba, pos_label=1)
    pr_auc = float(auc(recall_arr, precision_arr))

    # False Alarm Probability: fraction of true negatives predicted as positive
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fap = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "pr_auc": pr_auc,
        "false_alarm_probability": fap,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Build report text
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def build_report_text(
    metrics: dict,
    flag_counts: dict,
    n_total: int,
    n_planet: int,
    n_fp: int,
    model_path: Path,
) -> str:
    r = metrics["report"]
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    lines = [
        "=" * 72,
        "  EXOPLANET TRANSIT VETTER — MODEL EVALUATION REPORT",
        "=" * 72,
        "",
        f"  Model : {model_path}",
        f"  Eval set size : {n_total} samples  (Planet={n_planet}  False-Positive={n_fp})",
        "",
        "─" * 72,
        "  PER-CLASS METRICS",
        "─" * 72,
    ]

    for cls in ["FALSE POSITIVE", "CONFIRMED PLANET"]:
        d = r.get(cls, {})
        prec = d.get("precision", 0.0)
        rec  = d.get("recall",    0.0)
        f1   = d.get("f1-score",  0.0)
        sup  = int(d.get("support", 0))
        lines += [
            f"",
            f"  [{cls}]  (n={sup})",
            f"    Precision : {prec:.4f}  {_bar(prec)}",
            f"    Recall    : {rec:.4f}  {_bar(rec)}",
            f"    F1-Score  : {f1:.4f}  {_bar(f1)}",
        ]

    lines += [
        "",
        "─" * 72,
        "  OVERALL METRICS",
        "─" * 72,
        f"  Weighted F1           : {r['weighted avg']['f1-score']:.4f}",
        f"  Macro F1              : {r['macro avg']['f1-score']:.4f}",
        f"  PR-AUC (planet class) : {metrics['pr_auc']:.4f}",
        f"  False Alarm Prob (FAP): {metrics['false_alarm_probability']:.4f}",
        "",
        "─" * 72,
        "  CONFUSION MATRIX   (rows=True, cols=Predicted)",
        "─" * 72,
        "                   Pred=FP   Pred=Planet",
        f"  True=FP       :  {tn:6d}      {fp:6d}",
        f"  True=Planet   :  {fn:6d}      {tp:6d}",
        "",
        "  Interpretation:",
        f"    True Positives  (TP) : {tp}   ← real planets correctly confirmed",
        f"    True Negatives  (TN) : {tn}   ← false-positives correctly rejected",
        f"    False Positives (FP) : {fp}   ← false-positives mistaken for planets",
        f"    False Negatives (FN) : {fn}   ← real planets missed",
        "",
        "─" * 72,
        "  ASTROPHYSICAL SAFETY FLAGS  (triggered counts across eval set)",
        "─" * 72,
    ]

    if flag_counts:
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {flag:<45s} : {count}")
    else:
        lines.append("    No safety flags triggered.")

    lines += [
        "",
        "=" * 72,
        "  END OF REPORT",
        "=" * 72,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Evaluation set
    df = build_eval_set()
    X      = df[FEATURE_COLUMNS]
    y_true = df["label"].values.astype(int)
    n_total  = len(df)
    n_planet = int((y_true == 1).sum())
    n_fp     = int((y_true == 0).sum())

    # 2. Inference
    model = load_model()
    raw_preds, proba = infer(model, X)

    # 3. Safety layer
    final_preds, flag_counts = apply_safety_layer(df, raw_preds, proba)

    # 4. Metrics
    metrics = compute_metrics(y_true, final_preds, proba)

    # 5. Report
    report_text = build_report_text(
        metrics, flag_counts, n_total, n_planet, n_fp, MODEL_PATH
    )
    out_path = ROOT / "model_evaluation_metrics.txt"
    out_path.write_text(report_text, encoding="utf-8")

    # One-line summary
    wf1 = metrics["report"]["weighted avg"]["f1-score"]
    pr_auc = metrics["pr_auc"]
    fap    = metrics["false_alarm_probability"]
    summary = json.dumps({
        "weighted_f1": round(wf1, 4),
        "pr_auc":      round(pr_auc, 4),
        "fap":         round(fap, 4),
        "report_saved": str(out_path),
    })
    print(summary)


if __name__ == "__main__":
    main()
