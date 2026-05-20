"""Statistical A/B testing framework for model variant comparison."""
from typing import Dict, Any, Tuple, Optional
import json
from datetime import datetime

import numpy as np
from scipy import stats


def compute_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """
    Power analysis to determine required sample size.
    Production A/B tests run at n=12,000, 80% power, α=0.05.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    p_avg = (p1 + p2) / 2

    n = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = n / (p2 - p1) ** 2
    return int(np.ceil(n))


def run_proportion_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two-proportion z-test for comparing model variants on a binary outcome."""
    p_control = control_conversions / control_n
    p_treatment = treatment_conversions / treatment_n
    p_pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / control_n + 1 / treatment_n))
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    ci_delta = 1.96 * np.sqrt(
        p_control * (1 - p_control) / control_n + p_treatment * (1 - p_treatment) / treatment_n
    )
    delta = p_treatment - p_control

    return {
        "p_control": round(p_control, 5),
        "p_treatment": round(p_treatment, 5),
        "absolute_lift": round(delta, 5),
        "relative_lift_pct": round(100 * delta / p_control, 2),
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
        "confidence_interval_95": (round(delta - ci_delta, 5), round(delta + ci_delta, 5)),
    }


def run_metric_ttest(
    control_scores: np.ndarray,
    treatment_scores: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Welch t-test for continuous metrics (e.g. AUC, precision, recall)."""
    t_stat, p_value = stats.ttest_ind(treatment_scores, control_scores, equal_var=False)
    delta = treatment_scores.mean() - control_scores.mean()
    ci = stats.t.ppf(0.975, df=len(control_scores) + len(treatment_scores) - 2)
    se = np.sqrt(treatment_scores.var() / len(treatment_scores) + control_scores.var() / len(control_scores))

    return {
        "control_mean": round(float(control_scores.mean()), 5),
        "treatment_mean": round(float(treatment_scores.mean()), 5),
        "delta": round(float(delta), 5),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < alpha,
        "confidence_interval_95": (round(float(delta - ci * se), 5), round(float(delta + ci * se), 5)),
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
    """
    Full A/B test comparing two model variants via bootstrap AUC sampling.
    Mirrors the structured A/B experiments (n=12,000, 80% power, α=0.05) run in production.
    """
    rng = np.random.RandomState(random_state)
    auc_a_boots, auc_b_boots = [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() < 2:
            continue
        from sklearn.metrics import roc_auc_score
        auc_a_boots.append(roc_auc_score(y_true[idx], model_a_proba[idx]))
        auc_b_boots.append(roc_auc_score(y_true[idx], model_b_proba[idx]))

    auc_a = np.array(auc_a_boots)
    auc_b = np.array(auc_b_boots)
    ttest_result = run_metric_ttest(auc_a, auc_b, alpha=alpha)

    report = {
        "experiment": experiment_name,
        "timestamp": datetime.utcnow().isoformat(),
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "auc_comparison": ttest_result,
        "recommendation": "deploy model_b" if (ttest_result["delta"] > 0 and ttest_result["significant"]) else "keep model_a",
    }

    print(json.dumps(report, indent=2))
    return report
