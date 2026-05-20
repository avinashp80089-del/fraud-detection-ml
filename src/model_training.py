import time
from typing import Dict, Any, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# all six models benchmarked in prod — xgb won on AUC and inference speed
BENCHMARK_MODELS = {
    "xgboost": xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
        eval_metric="auc", random_state=42, n_jobs=-1,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    ),
    "logistic_regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, C=0.1, random_state=42,
    ),
    "neural_net": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        max_iter=200, random_state=42,
    ),
}


def benchmark_algorithms(
    X_train, X_test, y_train, y_test,
    experiment_name: str = "fraud-detection-benchmark",
) -> Dict[str, Dict[str, Any]]:
    mlflow.set_experiment(experiment_name)
    results = {}

    for name, model in BENCHMARK_MODELS.items():
        with mlflow.start_run(run_name=name):
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            proba = model.predict_proba(X_test)[:, 1]
            inf_ms = (time.perf_counter() - t0) * 1000

            auc = roc_auc_score(y_test, proba)
            ap = average_precision_score(y_test, proba)

            mlflow.log_params({"model": name})
            mlflow.log_metrics({
                "auc_roc": round(auc, 4),
                "avg_precision": round(ap, 4),
                "train_time_s": round(train_time, 3),
                "inference_time_ms": round(inf_ms, 2),
            })
            mlflow.sklearn.log_model(model, artifact_path=name)

            results[name] = {
                "model": model,
                "auc_roc": auc,
                "avg_precision": ap,
                "train_time_s": train_time,
                "inference_time_ms": inf_ms,
            }
            print(f"[{name}] AUC={auc:.4f}  AP={ap:.4f}  inf={inf_ms:.1f}ms")

    best = max(results, key=lambda k: results[k]["auc_roc"])
    print(f"\nBest: {best} (AUC={results[best]['auc_roc']:.4f})")
    return results


def tune_xgboost(
    X_train, y_train,
    n_trials: int = 50,
    experiment_name: str = "fraud-xgboost-tuning",
) -> Dict[str, Any]:
    mlflow.set_experiment(experiment_name)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 50),
            "eval_metric": "auc", "random_state": 42, "n_jobs": -1,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train)
        return roc_auc_score(y_train, m.predict_proba(X_train)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    with mlflow.start_run(run_name="xgboost_best"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_auc", study.best_value)

    print(f"Best AUC: {study.best_value:.4f}")
    return study.best_params


def train_production_model(
    X_train, y_train,
    params: Dict[str, Any] = None,
    experiment_name: str = "fraud-detection-production",
    run_name: str = "xgboost_production",
) -> Tuple[xgb.XGBClassifier, str]:
    if params is None:
        params = {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 30,
            "eval_metric": "auc", "random_state": 42, "n_jobs": -1,
        }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        mlflow.log_params(params)
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id
        print(f"Logged run: {run_id}")

    return model, run_id
