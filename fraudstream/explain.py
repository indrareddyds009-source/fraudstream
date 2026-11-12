"""Reason codes and analyst-facing explanations, grounded in the model's own feature contributions.

LightGBM's exact TreeSHAP contributions (pred_contrib) give each feature's push on the score.
The top positive contributors become reason codes. An optional LLM rewrites them as a short
case note, but only from those facts, and the note is rejected (falling back to the template)
if it mentions any number that is not in the facts.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from .features import CATEGORY_CODES, FEATURE_LABELS, FEATURES

CATEGORY_NAMES = {v: k.replace("_", " ") for k, v in CATEGORY_CODES.items()}


def _fmt(feature: str, v: float) -> str:
    if feature in ("amount", "log_amount", "amount_1h"):
        return f"${(np.expm1(v) if feature == 'log_amount' else v):,.2f}"
    if feature == "dist_home_km":
        return f"{v:,.0f} km"
    if feature == "speed_kmh":
        return f"{v:,.0f} km/h"
    if feature == "secs_since_last":
        return f"{v / 60:,.1f} min" if v < 7200 else f"{v / 3600:,.1f} h"
    if feature == "category":
        return CATEGORY_NAMES.get(int(v), "unknown")
    if feature == "hour":
        return f"{int(v):02d}:00"
    if feature == "amount_z":
        return f"{v:.1f} std devs above usual" if v >= 0 else f"{v:.1f} std devs"
    if feature in ("amount_vs_max",):
        return f"{v:.1f}× previous max"
    if feature == "cat_share":
        return f"{v:.0%} of past purchases"
    if feature in ("new_merchant", "new_device", "online", "is_night"):
        return "yes" if v >= 0.5 else "no"
    return f"{v:g}"


def reason_codes(booster, x: pd.DataFrame, top_k: int = 3) -> list[dict]:
    contrib = booster.predict(x[FEATURES], pred_contrib=True)[0][:-1]   # last column = bias
    order = np.argsort(-contrib)
    out = []
    for i in order[:top_k]:
        if contrib[i] <= 0:
            break
        f = FEATURES[i]
        out.append({"feature": f, "label": FEATURE_LABELS[f], "value": _fmt(f, float(x.iloc[0][f])),
                     "contribution": round(float(contrib[i]), 3)})
    return out


def template_note(score: float, decision: str, reasons: list[dict]) -> str:
    if not reasons:
        return f"Risk score {score:.2f} ({decision}). No individual signal stands out."
    parts = "; ".join(f"{r['label']}: {r['value']}" for r in reasons)
    return f"Risk score {score:.2f} ({decision}). Main drivers: {parts}."


def llm_note(score: float, decision: str, reasons: list[dict]) -> str:
    """Optional: GPT-written note, grounded and number-checked. Falls back to the template."""
    fallback = template_note(score, decision, reasons)
    if not os.getenv("OPENAI_API_KEY"):
        return fallback
    from openai import OpenAI

    facts = "\n".join(f"- {r['label']}: {r['value']}" for r in reasons)
    prompt = (f"Write a 2-sentence note for a fraud analyst. Use ONLY these facts; do not add numbers.\n"
              f"Risk score: {score:.2f}\nDecision: {decision}\nSignals:\n{facts}")
    try:
        text = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0,
            messages=[{"role": "user", "content": prompt}]).choices[0].message.content.strip()
    except Exception:  # noqa: BLE001  (network/auth errors must not break scoring)
        return fallback
    return text if is_grounded(text, prompt) else fallback


def is_grounded(text: str, facts: str) -> bool:
    """Every number in the generated text must appear in the facts."""
    return _numbers(text) <= _numbers(facts)


def _numbers(s: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", s)}
