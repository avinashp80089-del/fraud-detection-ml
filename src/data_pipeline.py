import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


FRAUD_RATE = 0.003


def load_transaction_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.dropna(subset=["amount", "user_id", "timestamp"])
    df = df[df["amount"] > 0]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["user_id", "timestamp"])

    # velocity signals — top SHAP features by a wide margin
    df["txn_count_1h"] = (
        df.groupby("user_id")["timestamp"]
        .transform(lambda s: s.expanding().count())
    )
    df["amount_sum_24h"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: x.rolling(window=24, min_periods=1).sum())
    )
    df["amount_mean_7d"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: x.rolling(window=168, min_periods=1).mean())
    )
    df["amount_std_7d"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: x.rolling(window=168, min_periods=1).std().fillna(0))
    )

    ts = df["timestamp"]
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)).astype(int)

    df["amount_log"] = np.log1p(df["amount"])
    df["amount_zscore"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    )

    df["velocity_x_amount"] = df["txn_count_1h"] * df["amount_log"]
    return df


FEATURE_COLS = [
    "amount", "amount_log", "amount_zscore",
    "txn_count_1h", "amount_sum_24h", "amount_mean_7d", "amount_std_7d",
    "hour_of_day", "day_of_week", "is_weekend", "is_night",
    "velocity_x_amount",
]


def handle_class_imbalance(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    smote = SMOTE(sampling_strategy=0.1, random_state=random_state, k_neighbors=5)
    return smote.fit_resample(X, y)


def prepare_dataset(
    df: pd.DataFrame,
    label_col: str = "is_fraud",
    test_size: float = 0.2,
    apply_smote: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    df = engineer_features(df)
    features = [c for c in FEATURE_COLS if c in df.columns]

    X = df[features].fillna(0).values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if apply_smote:
        X_train, y_train = handle_class_imbalance(X_train, y_train, random_state)

    return X_train, X_test, y_train, y_test, scaler


def generate_sample_data(n_samples: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    n_users = 5_000
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="1min")

    df = pd.DataFrame({
        "transaction_id": [f"txn_{i:07d}" for i in range(n_samples)],
        "user_id": rng.randint(0, n_users, n_samples),
        "amount": np.abs(rng.lognormal(mean=4.0, sigma=1.5, size=n_samples)),
        "timestamp": timestamps,
        "merchant_category": rng.choice(["crypto", "retail", "transfer", "atm"], n_samples),
        "country_code": rng.choice(
            ["US", "UK", "NG", "RU", "CN", "DE"], n_samples,
            p=[0.5, 0.15, 0.1, 0.1, 0.1, 0.05]
        ),
    })

    n_fraud = int(n_samples * FRAUD_RATE)
    fraud_idx = rng.choice(n_samples, n_fraud, replace=False)
    df["is_fraud"] = 0
    df.loc[fraud_idx, "is_fraud"] = 1
    df.loc[fraud_idx, "amount"] *= rng.uniform(5, 20, n_fraud)

    return df.sort_values("timestamp").reset_index(drop=True)
