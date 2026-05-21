import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def compute_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    p_avg = (p1 + p2) / 2

    n = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(n / (p2 - p1) ** 2))


def run_proportion_test(
    control_conversions: int, control_n: int,
    treatment_conversions: int, treatment_n: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    p_c = control_conversions / control_n
    p_t = treatment_conversions / treatment_n
    p_pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / control_n + 1 / treatment_n))
    z = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    delta = p_t - p_c
    ci_margin = 1.96 * np.sqrt(p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n)

    return {
        "p_control": round(p_c, 5),
        "p_treatment": round(p_t, 5),
        "absolute_lift": round(delta, 5),
        "relative_lift_pct": round(100 * delta / p_c, 2),
        "z_statistic": round(z, 4),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "confidence_interval_95": (round(delta - ci_margin, 5), round(delta + ci_margin, 5)),
    }


def run_metric_ttest(
    control_scores: np.ndarray,
    treatment_scores: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    t_stat, p_value = stats.ttest_ind(treatment_scores, control_scores, equal_var=False)
    delta = treatment_scores.mean() - control_scores.mean()
    df = len(control_scores) + len(treatment_scores) - 2
    se = np.sqrt(treatment_scores.var() / len(treatment_scores) + control_scores.var() / len(control_scores))
    ci = stats.t.ppf(0.975, df=df) * se

    return {
        "control_mean": round(float(control_scores.mean()), 5),
        "treatment_mean": round(float(treatment_scores.mean()), 5),
        "delta": round(float(delta), 5),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "confidence_interval_95": (round(float(delta - ci), 5), round(float(delta + ci), 5)),
    }


def run_model_ab_test(
    model_a_proba: np.ndarray,
    model_b_proba: np.ndarray,
    y_true: np.ndarray,
    experiment_name: str = "model_ab_test",
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    rng = np.random.RandomState(random_state)
    auc_a_boots, auc_b_boots = [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() < 2:
            continue
        auc_a_boots.append(roc_auc_score(y_true[idx], model_a_proba[idx]))
        auc_b_boots.append(roc_auc_score(y_true[idx], model_b_proba[idx]))

    auc_a, auc_b = np.array(auc_a_boots), np.array(auc_b_boots)
    ttest = run_metric_ttest(auc_a, auc_b, alpha=alpha)

    report = {
        "experiment": experiment_name,
        "timestamp": datetime.utcnow().isoformat(),
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "auc_comparison": ttest,
        "recommendation": (
            "deploy model_b" if (ttest["delta"] > 0 and ttest["significant"])
            else "keep model_a"
        ),
    }
    print(json.dumps(report, indent=2))
    return report
