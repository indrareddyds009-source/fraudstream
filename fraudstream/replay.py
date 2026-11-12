"""Replay a slice of the stream through the live scorer and report latency, throughput and decisions.

    python -m fraudstream.replay --from-day 50 --to-day 60
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .scorer import FraudScorer
from .simulate import START, simulate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--from-day", type=int, default=50)
    ap.add_argument("--to-day", type=int, default=60)
    ap.add_argument("--limit", type=int, default=20000)
    a = ap.parse_args(argv)
    df = simulate()
    day = (df["ts"] - START).dt.days
    sc = FraudScorer(a.artifacts)
    sc.warm(df[day < a.from_day].to_dict("records"))
    stream = df[(day >= a.from_day) & (day < a.to_day)].head(a.limit)
    lat, decisions, hits = [], {"approve": 0, "review": 0, "decline": 0}, 0
    t0 = time.perf_counter()
    example = None
    for t in stream.to_dict("records"):
        r = sc.score(t)
        if example is None and r["decision"] != "approve" and t["is_fraud"]:
            example = sc.explain(t["txn_id"])
        lat.append(r["latency_ms"])
        decisions[r["decision"]] += 1
        hits += int(r["decision"] != "approve" and t["is_fraud"] == 1)
    wall = time.perf_counter() - t0
    n_fraud = int(stream["is_fraud"].sum())
    print(f"scored {len(stream):,} txns in {wall:.1f}s → {len(stream) / wall:,.0f} txn/s (single process)")
    print(f"latency p50 {np.percentile(lat, 50):.2f} ms · p99 {np.percentile(lat, 99):.2f} ms")
    print(f"decisions {decisions} · fraud flagged {hits}/{n_fraud}")
    if example:
        print("\nexample explanation:", example["note"])
    print("\ndrift:", {k: v for k, v in sc.drift().items() if k != "psi"})


if __name__ == "__main__":
    main()
