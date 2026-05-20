import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Fraud Detection API", version="1.0.0")

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


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_tier: str
    inference_ms: float


def load_model(model, scaler=None, threshold: float = 0.35):
    global _model, _scaler, _threshold
    _model, _scaler, _threshold = model, scaler, threshold


def _risk_tier(prob: float) -> str:
    if prob < 0.1:   return "low"
    if prob < 0.35:  return "medium"
    if prob < 0.7:   return "high"
    return "critical"


def _score(features: List[float]) -> float:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    arr = np.array(features).reshape(1, -1)
    if _scaler:
        arr = _scaler.transform(arr)
    return float(_model.predict_proba(arr)[0, 1])


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None, "threshold": _threshold}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest):
    global _request_count
    t0 = time.perf_counter()

    features = [
        req.amount, req.amount_log, req.amount_zscore,
        req.txn_count_1h, req.amount_sum_24h, req.amount_mean_7d,
        req.amount_std_7d, req.hour_of_day, req.day_of_week,
        req.is_weekend, req.is_night, req.velocity_x_amount,
    ]

    try:
        prob = _score(features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    _request_count += 1
    return PredictionResponse(
        transaction_id=req.transaction_id,
        fraud_probability=round(prob, 5),
        is_fraud=prob >= _threshold,
        risk_tier=_risk_tier(prob),
        inference_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(transactions: List[TransactionRequest]):
    return [predict(t) for t in transactions]
