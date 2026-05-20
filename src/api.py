"""FastAPI scoring endpoint — mirrors the production REST API on SageMaker."""
from typing import List, Optional
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring at p99 <180ms. Deployed on SageMaker.",
    version="1.0.0",
)

_model = None
_scaler = None
_threshold = 0.35
_request_count = 0


class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float = Field(..., gt=0)
    amount_log: float
    amount_zscore: float
    txn_count_1h: int = Field(..., ge=0)
    amount_sum_24h: float
    amount_mean_7d: float
    amount_std_7d: float
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_night: int = Field(..., ge=0, le=1)
    velocity_x_amount: float


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_tier: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float
    requests_served: int


def _risk_tier(prob: float) -> str:
    if prob < 0.1:
        return "low"
    if prob < 0.35:
        return "medium"
    if prob < 0.7:
        return "high"
    return "critical"


def _score_features(features: List[float]) -> float:
    """Score a single feature vector — requires model to be loaded via load_model()."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    arr = np.array(features).reshape(1, -1)
    if _scaler is not None:
        arr = _scaler.transform(arr)
    return float(_model.predict_proba(arr)[0, 1])


def load_model(model, scaler=None, threshold: float = 0.35):
    """Inject model and scaler into the API (called at server startup)."""
    global _model, _scaler, _threshold
    _model = model
    _scaler = scaler
    _threshold = threshold


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        threshold=_threshold,
        requests_served=_request_count,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    global _request_count
    t0 = time.perf_counter()

    features = [
        request.amount, request.amount_log, request.amount_zscore,
        request.txn_count_1h, request.amount_sum_24h, request.amount_mean_7d,
        request.amount_std_7d, request.hour_of_day, request.day_of_week,
        request.is_weekend, request.is_night, request.velocity_x_amount,
    ]

    try:
        prob = _score_features(features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    _request_count += 1
    inference_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(prob, 5),
        is_fraud=prob >= _threshold,
        risk_tier=_risk_tier(prob),
        inference_ms=round(inference_ms, 2),
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(request: BatchRequest):
    return [predict(txn) for txn in request.transactions]
