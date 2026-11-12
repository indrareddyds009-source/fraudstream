"""Training, cost-based thresholding, evaluation, and the model artifact."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import FEATURES

REVIEW_COST = 5.0      # analyst time / customer friction per alert (USD)
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "scale_pos_weight": 10, "verbose": -1, "seed": 7}


@dataclass
class ModelCard:
    version: str
    trained_on: str
    threshold: float
    review_cost: float
    features: list
    valid_metrics: dict
    reference: dict    # per-feature PSI bin edges + expected proportions, plus score distribution


def fit(X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None, rounds: int = 400) -> lgb.Booster:
    dtrain = lgb.Dataset(X[FEATURES], y, categorical_feature=["category"], free_raw_data=False)
    valid = [lgb.Dataset(X_val[FEATURES], y_val, reference=dtrain)] if X_val is not None else []
    cbs = [lgb.early_stopping(40, verbose=False)] if valid else []
    return lgb.train(PARAMS, dtrain, num_boost_round=rounds, valid_sets=valid, callbacks=cbs)


def choose_threshold(scores: np.ndarray, y: np.ndarray, amounts: np.ndarray, review_cost: float = REVIEW_COST):
    """Pick the threshold that minimizes expected cost = missed fraud dollars + review cost per alert."""
    best = (np.inf, 0.5)
    for thr in np.unique(np.quantile(scores, np.linspace(0.90, 0.9999, 400))):
        flag = scores >= thr
        cost = amounts[(~flag) & (y == 1)].sum() + review_cost * flag.sum()
        if cost < best[0]:
            best = (cost, float(thr))
    return best[1]


def evaluate(scores, y, amounts, thr) -> dict:
    flag = scores >= thr
    tp = int((flag & (y == 1)).sum())
    return {"n": len(y), "fraud_rate": float(y.mean()),
            "pr_auc": float(average_precision_score(y, scores)) if y.sum() else None,
            "roc_auc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else None,
            "recall": tp / max(int(y.sum()), 1), "precision": tp / max(int(flag.sum()), 1),
            "fraud_dollars_caught": float(amounts[flag & (y == 1)].sum() / max(amounts[y == 1].sum(), 1e-9)),
            "alerts_per_1k": float(flag.mean() * 1000),
            "expected_cost": float(amounts[(~flag) & (y == 1)].sum() + REVIEW_COST * flag.sum())}


def rules_baseline(X: pd.DataFrame) -> np.ndarray:
    """What many teams start with: hand-written rules. Used as a benchmark."""
    return ((X["amount"] > 1000) | ((X["new_device"] == 1) & (X["dist_home_km"] > 500))
            | (X["small_10m"] >= 3) | (X["speed_kmh"] > 900)).astype(float).to_numpy()


def save(booster: lgb.Booster, card: ModelCard, out_dir: str):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(p / "model.txt"))
    (p / "model_card.json").write_text(json.dumps(asdict(card), indent=2))


def load(out_dir: str) -> tuple[lgb.Booster, ModelCard]:
    p = Path(out_dir)
    return lgb.Booster(model_file=str(p / "model.txt")), ModelCard(**json.loads((p / "model_card.json").read_text()))
