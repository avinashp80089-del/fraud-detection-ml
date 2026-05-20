import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def evaluate_at_threshold(y_true, proba, threshold: float) -> Dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, proba), 4),
        "avg_precision": round(average_precision_score(y_true, proba), 4),
    }


def calibrate_threshold(
    y_true, proba, target_precision: float = 0.85
) -> Tuple[float, Dict[str, float]]:
    """Find threshold that hits target precision while maximising recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)

    best_thr, best_rec = 0.5, 0.0
    for prec, rec, thr in zip(precisions[:-1], recalls[:-1], thresholds):
        if prec >= target_precision and rec > best_rec:
            best_rec = rec
            best_thr = thr

    metrics = evaluate_at_threshold(y_true, proba, best_thr)
    print(f"threshold={best_thr:.3f}  precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}")
    return best_thr, metrics


def full_evaluation_report(
    model, X_test, y_test,
    threshold: Optional[float] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    proba = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        threshold, _ = calibrate_threshold(y_test, proba)

    metrics = evaluate_at_threshold(y_test, proba, threshold)
    y_pred = (proba >= threshold).astype(int)

    report = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.log_dict(report, "evaluation_report.json")

    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))
    return report


def shap_analysis(model, X, feature_names: List[str], top_n: int = 10) -> Dict[str, float]:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    importance = dict(zip(feature_names, np.abs(shap_vals).mean(axis=0).tolist()))
    ranked = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n])

    print("Top SHAP features:")
    for feat, val in ranked.items():
        print(f"  {feat:<30} {val:.4f}")

    return ranked


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """PSI-based drift detection — flag features that shifted since last training window."""
    report = {"timestamp": datetime.utcnow().isoformat(), "features": {}, "drifted": []}

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = _psi(reference_df[col].dropna().values, current_df[col].dropna().values)
        report["features"][col] = {"psi": round(psi, 4), "drifted": psi > threshold}
        if psi > threshold:
            report["drifted"].append(col)

    if report["drifted"]:
        print(f"DRIFT in: {report['drifted']} — retrain on rolling 30-day window")
    else:
        print("No drift detected.")

    return report


def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf

    ref_pct = (np.histogram(reference, bins=bins)[0] + 1e-6) / len(reference)
    cur_pct = (np.histogram(current, bins=bins)[0] + 1e-6) / len(current)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
