"""Real-time scorer: online features → model → decision → reason codes, with a rolling drift window."""
from __future__ import annotations

import time
from collections import OrderedDict, deque

import numpy as np
import pandas as pd

from . import explain, model, monitor
from .features import FEATURES, FeatureStore

DECLINE_SCORE = 0.8


class FraudScorer:
    def __init__(self, artifact_dir: str, window: int = 5000, keep: int = 10000):
        self.booster, self.card = model.load(artifact_dir)
        self.store = FeatureStore()
        self.recent_feats: deque = deque(maxlen=window)
        self.recent_scores: deque = deque(maxlen=window)
        self.cases: OrderedDict = OrderedDict()   # txn_id -> (features, score, decision) for /explain
        self.keep = keep

    def warm(self, history: list[dict]) -> None:
        """Load card history into the feature store (in production this state lives in Redis or similar)."""
        for t in history:
            self.store.update(t)

    def score(self, txn: dict) -> dict:
        t0 = time.perf_counter()
        f = self.store.process(txn)
        s = float(self.booster.predict(np.array([[f[k] for k in FEATURES]], dtype=float))[0])
        decision = "decline" if s >= max(DECLINE_SCORE, self.card.threshold) else \
            "review" if s >= self.card.threshold else "approve"
        reasons = explain.reason_codes(self.booster, pd.DataFrame([f])[FEATURES]) if decision != "approve" else []
        self.recent_feats.append(f)
        self.recent_scores.append(s)
        self.cases[txn["txn_id"]] = (f, s, decision)
        if len(self.cases) > self.keep:
            self.cases.popitem(last=False)
        return {"txn_id": txn["txn_id"], "score": round(s, 4), "decision": decision, "reasons": reasons,
                "model_version": self.card.version, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    def explain(self, txn_id: str, use_llm: bool = False) -> dict | None:
        if txn_id not in self.cases:
            return None
        f, s, decision = self.cases[txn_id]
        reasons = explain.reason_codes(self.booster, pd.DataFrame([f])[FEATURES], top_k=4)
        note = explain.llm_note(s, decision, reasons) if use_llm else explain.template_note(s, decision, reasons)
        return {"txn_id": txn_id, "score": round(s, 4), "decision": decision, "reasons": reasons, "note": note}

    def drift(self) -> dict:
        if len(self.recent_scores) < 500:
            return {"status": "insufficient_data", "window": len(self.recent_scores)}
        X = pd.DataFrame(list(self.recent_feats))
        psi = monitor.drift_report(self.card.reference, X, np.asarray(self.recent_scores))
        worst = max((v, k) for k, v in psi.items())
        return {"window": len(self.recent_scores), "score_psi": round(psi["__score__"], 4),
                "status": monitor.status(worst[0]), "worst": {"feature": worst[1], "psi": round(worst[0], 4)},
                "psi": {k: round(v, 4) for k, v in psi.items()}}
