"""Drift monitoring with the Population Stability Index (PSI).

PSI < 0.1 stable · 0.1-0.25 moderate shift · > 0.25 significant shift (common industry rule of thumb).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES

PSI_WARN, PSI_ALERT = 0.1, 0.25


def reference_bins(values: np.ndarray, bins: int = 10) -> dict:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant / binary feature: use value bins
        edges = np.array([-np.inf, *np.unique(values)[:-1] + 1e-9, np.inf]) if len(np.unique(values)) > 1 \
            else np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return {"edges": edges.tolist(), "expected": _props(values, edges).tolist()}


def _props(values, edges):
    counts = np.histogram(values, bins=np.asarray(edges))[0]
    return counts / max(counts.sum(), 1)


def psi(ref: dict, values: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(ref["expected"]), eps, None)
    a = np.clip(_props(values, ref["edges"]), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference(X: pd.DataFrame, scores: np.ndarray) -> dict:
    ref = {f: reference_bins(X[f].to_numpy(dtype=float)) for f in FEATURES}
    ref["__score__"] = reference_bins(scores)
    return ref


def drift_report(reference: dict, X: pd.DataFrame, scores: np.ndarray) -> dict:
    out = {f: psi(reference[f], X[f].to_numpy(dtype=float)) for f in FEATURES}
    out["__score__"] = psi(reference["__score__"], scores)
    return out


def status(value: float) -> str:
    return "ALERT" if value > PSI_ALERT else "WARN" if value > PSI_WARN else "ok"


def performance_status(baseline_recall: float, recall: float, max_drop: float = 0.10) -> str:
    """Label-based check once delayed fraud labels (chargebacks) arrive."""
    return "ALERT" if baseline_recall - recall > max_drop else "ok"
