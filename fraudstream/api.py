"""FastAPI scoring service.

    python -m fraudstream.pipeline           # trains and writes artifacts/
    uvicorn fraudstream.api:app --port 8000
"""
from __future__ import annotations

import datetime as dt
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .scorer import FraudScorer

app = FastAPI(title="FraudStream", version="0.1.0")


@lru_cache
def scorer() -> FraudScorer:
    return FraudScorer(os.getenv("FRAUDSTREAM_ARTIFACTS", "artifacts"))


class Txn(BaseModel):
    txn_id: str
    ts: dt.datetime
    card_id: str
    merchant: str
    category: str
    amount: float = Field(gt=0)
    online: bool
    lat: float
    lon: float
    device_id: str | None = None
    home_lat: float
    home_lon: float


@app.get("/health")
def health():
    s = scorer()
    return {"status": "ok", "model_version": s.card.version, "threshold": s.card.threshold}


@app.post("/score")
def score(t: Txn):
    return scorer().score(t.model_dump())


@app.get("/explain/{txn_id}")
def explain(txn_id: str, llm: bool = False):
    out = scorer().explain(txn_id, use_llm=llm)
    if out is None:
        raise HTTPException(404, "unknown or expired txn_id")
    return out


@app.get("/monitor/drift")
def drift():
    return scorer().drift()
