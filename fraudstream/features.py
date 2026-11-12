"""Online feature store.

The SAME code computes features during offline training (by replaying history in time
order) and during live scoring, so there is no training/serving skew. Each feature only
uses information available before the transaction being scored.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .simulate import CATEGORIES, haversine_km

CATEGORY_CODES = {c: i for i, c in enumerate(CATEGORIES)}
FEATURES = ["amount", "log_amount", "amount_z", "amount_vs_max", "hour", "is_night", "online", "category",
            "dist_home_km", "speed_kmh", "secs_since_last", "txn_1h", "txn_24h", "amount_1h", "small_10m",
            "new_merchant", "new_device", "cat_share"]
FEATURE_LABELS = {
    "amount": "transaction amount", "log_amount": "transaction amount", "amount_z": "amount vs. card's usual",
    "amount_vs_max": "amount vs. card's largest previous purchase", "hour": "time of day", "is_night": "overnight timing",
    "online": "card-not-present", "category": "merchant category", "dist_home_km": "distance from home",
    "speed_kmh": "travel speed since last transaction", "secs_since_last": "time since previous transaction",
    "txn_1h": "transactions in the past hour", "txn_24h": "transactions in the past 24 hours",
    "amount_1h": "spend in the past hour", "small_10m": "small 'test' charges in 10 minutes",
    "new_merchant": "first purchase at this merchant", "new_device": "unrecognized device",
"cat_share": "how often this card uses this category"}


@dataclass
class CardState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    max_amount: float = 0.0
    last_ts: float | None = None
    last_ll: tuple | None = None
    recent: deque = field(default_factory=deque)       # (ts, amount) within 24h
    merchants: set = field(default_factory=set)
    devices: set = field(default_factory=set)
    cat_counts: dict = field(default_factory=dict)


class FeatureStore:
    def __init__(self):
        self.cards: dict[str, CardState] = {}

    def compute(self, t: dict) -> dict:
        """Features for transaction t using only prior state. Call update(t) afterwards."""
        s = self.cards.get(t["card_id"]) or CardState()
        ts = t["ts"].timestamp() if hasattr(t["ts"], "timestamp") else float(t["ts"])
        amt = float(t["amount"])
        std = math.sqrt(s.m2 / (s.n - 1)) if s.n > 1 else 0.0
        ll = (t["lat"], t["lon"])
        dist_home = haversine_km(ll, (t["home_lat"], t["home_lon"]))
        if s.last_ts is not None:
            gap = max(ts - s.last_ts, 1.0)
            speed = haversine_km(ll, s.last_ll) / (gap / 3600)
        else:
            gap, speed = 30 * 86400.0, 0.0
        recent = [(x, a) for x, a in s.recent if ts - x <= 86400]
        hour = int(t["ts"].hour) if hasattr(t["ts"], "hour") else 12
        dev = t.get("device_id")
        return {
            "amount": amt, "log_amount": math.log1p(amt),
            "amount_z": (amt - s.mean) / std if std > 0 else 0.0,
            "amount_vs_max": amt / s.max_amount if s.max_amount > 0 else 1.0,
            "hour": hour, "is_night": int(hour < 5), "online": int(t["online"]),
            "category": CATEGORY_CODES.get(t["category"], -1),
            "dist_home_km": dist_home, "speed_kmh": min(speed, 5000.0), "secs_since_last": min(gap, 30 * 86400.0),
            "txn_1h": sum(1 for x, _ in recent if ts - x <= 3600),
            "txn_24h": len(recent),
            "amount_1h": sum(a for x, a in recent if ts - x <= 3600),
            "small_10m": sum(1 for x, a in recent if ts - x <= 600 and a < 5),
            "new_merchant": int(t["merchant"] not in s.merchants and s.n > 0),
            "new_device": int(bool(dev) and dev not in s.devices and len(s.devices) > 0),
            "cat_share": s.cat_counts.get(t["category"], 0) / s.n if s.n else 0.0,
        }

    def update(self, t: dict) -> None:
        s = self.cards.setdefault(t["card_id"], CardState())
        ts = t["ts"].timestamp() if hasattr(t["ts"], "timestamp") else float(t["ts"])
        amt = float(t["amount"])
        s.n += 1
        d = amt - s.mean
        s.mean += d / s.n
        s.m2 += d * (amt - s.mean)
        s.max_amount = max(s.max_amount, amt)
        s.last_ts, s.last_ll = ts, (t["lat"], t["lon"])
        s.recent.append((ts, amt))
        while s.recent and ts - s.recent[0][0] > 86400:
            s.recent.popleft()
        s.merchants.add(t["merchant"])
        if t.get("device_id"):
            s.devices.add(t["device_id"])
        s.cat_counts[t["category"]] = s.cat_counts.get(t["category"], 0) + 1

    def process(self, t: dict) -> dict:
        f = self.compute(t)
        self.update(t)
        return f


def build_feature_frame(df):
    """Replay a time-ordered transaction frame through the store (used for training)."""
    import pandas as pd

    store = FeatureStore()
    feats = [store.process(t) for t in df.to_dict("records")]
    return pd.DataFrame(feats, index=df.index)[FEATURES], store
