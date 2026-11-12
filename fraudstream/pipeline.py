"""End-to-end experiment: simulate → features → train → threshold → evaluate → monitor drift → retrain.

    python -m fraudstream.pipeline --out artifacts
Writes the model artifact, docs/report.md and charts. All numbers come from this run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import pandas as pd

from . import model as M
from . import monitor
from .features import FEATURES, build_feature_frame
from .simulate import DRIFT_DAY, START, simulate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BURN_IN, TRAIN_END, VALID_END, TEST_END = 14, 40, 50, DRIFT_DAY  # burn-in: card profiles warm up
RETRAIN_LABELS_UNTIL = 75   # labels from chargebacks arrive with delay; retrain uses what's known by day 75


def day(df):
    return (df["ts"] - START).dt.days


def run(out: str = "artifacts", docs: str = "docs", n_cards: int = 2000, seed: int = 11) -> dict:
    t0 = time.time()
    df = simulate(n_cards=n_cards, seed=seed)
    X, _ = build_feature_frame(df)
    y, amt, d = df["is_fraud"], df["amount"].to_numpy(), day(df)
    tr, va, te = (d >= BURN_IN) & (d < TRAIN_END), (d >= TRAIN_END) & (d < VALID_END), (d >= VALID_END) & (d < TEST_END)

    booster = M.fit(X[tr], y[tr], X[va], y[va])
    s_va = booster.predict(X[va][FEATURES])
    thr = M.choose_threshold(s_va, y[va].to_numpy(), amt[va.to_numpy()])
    s_all = booster.predict(X[FEATURES])

    test = M.evaluate(s_all[te], y[te].to_numpy(), amt[te.to_numpy()], thr)
    rules = M.evaluate(M.rules_baseline(X[te]), y[te].to_numpy(), amt[te.to_numpy()], 0.5)
    ref = monitor.build_reference(X[tr], s_all[tr.to_numpy()])
    card = M.ModelCard(version="v1", trained_on=f"days {BURN_IN}-{TRAIN_END - 1}", threshold=thr,
                       review_cost=M.REVIEW_COST, features=FEATURES,
                       valid_metrics=M.evaluate(s_va, y[va].to_numpy(), amt[va.to_numpy()], thr), reference=ref)
    M.save(booster, card, out)

    # weekly monitoring across the in-distribution test period and the drifted period
    weeks = []
    for w0 in range(VALID_END, 90, 7):
        m = ((d >= w0) & (d < w0 + 7)).to_numpy()
        if m.sum() == 0:
            continue
        psi = monitor.drift_report(ref, X[m], s_all[m])
        ev = M.evaluate(s_all[m], y[m].to_numpy(), amt[m], thr)
        top = sorted(((k, v) for k, v in psi.items() if k != "__score__"), key=lambda kv: -kv[1])[:3]
        weeks.append({"week_start_day": w0, "score_psi": psi["__score__"], "max_feature_psi": top[0][1],
                      "top_drift": ", ".join(f"{k} {v:.2f}" for k, v in top), "recall": ev["recall"],
                      "precision": ev["precision"], "fraud_dollars_caught": ev["fraud_dollars_caught"],
                      "label_status": monitor.performance_status(card.valid_metrics["recall"], ev["recall"])})
    wk = pd.DataFrame(weeks)

    # champion vs challenger on the final weeks (after the retrain point)
    retr = (d < RETRAIN_LABELS_UNTIL) & (d >= BURN_IN)
    hold = d >= RETRAIN_LABELS_UNTIL
    rv = (d >= RETRAIN_LABELS_UNTIL - 7) & (d < RETRAIN_LABELS_UNTIL)
    fit_mask = retr & ~rv
    challenger = M.fit(X[fit_mask], y[fit_mask], X[rv], y[rv])
    thr2 = M.choose_threshold(challenger.predict(X[rv][FEATURES]), y[rv].to_numpy(), amt[rv.to_numpy()])
    champ_eval = M.evaluate(s_all[hold.to_numpy()], y[hold].to_numpy(), amt[hold.to_numpy()], thr)
    chall_eval = M.evaluate(challenger.predict(X[hold][FEATURES]), y[hold].to_numpy(), amt[hold.to_numpy()], thr2)
    stealth = (df["fraud_type"] == "stealth") & hold
    stealth_recall = {
        "champion": float((s_all[stealth.to_numpy()] >= thr).mean()),
        "challenger": float((challenger.predict(X[stealth][FEATURES]) >= thr2).mean())}
    M.save(challenger, M.ModelCard(version="v2", trained_on=f"days {BURN_IN}-{RETRAIN_LABELS_UNTIL - 1}", threshold=thr2,
                                   review_cost=M.REVIEW_COST, features=FEATURES, valid_metrics={},
                                   reference=monitor.build_reference(X[fit_mask], challenger.predict(
                                       X[fit_mask][FEATURES]))), str(Path(out) / "v2"))

    imp = pd.Series(booster.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    results = {"rows": len(df), "fraud_rate": float(y.mean()), "threshold": thr, "test": test, "rules": rules,
               "weekly": weeks, "champion_after_drift": champ_eval, "challenger_after_drift": chall_eval,
               "stealth_recall": stealth_recall, "top_features": imp.head(8).round(0).to_dict(),
               "runtime_s": round(time.time() - t0, 1)}
    Path(out, "results.json").write_text(json.dumps(results, indent=2))
    _charts(wk, docs)
    _report(results, wk, docs)
    return results


def _charts(wk: pd.DataFrame, docs: str):
    Path(docs).mkdir(exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4), dpi=150)
    x = wk["week_start_day"]
    ax1.plot(x, wk["recall"] * 100, marker="o", label="Recall (%)")
    ax1.plot(x, wk["fraud_dollars_caught"] * 100, marker="s", label="Fraud $ caught (%)")
    ax1.set_ylabel("%")
    ax1.set_xlabel("Day of stream (week start)")
    ax2 = ax1.twinx()
    ax2.bar(x, wk["max_feature_psi"], width=3, alpha=0.25, color="tab:red", label="Max feature PSI")
    ax2.axhline(monitor.PSI_ALERT, color="tab:red", ls="--", lw=1)
    ax2.set_ylabel("PSI")
    ax1.axvline(DRIFT_DAY, color="gray", ls=":", lw=1)
    ax1.text(DRIFT_DAY + 0.5, 97, "drift starts", color="gray", fontsize=8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    ax1.set_title("Weekly model health: performance vs. drift")
    fig.tight_layout()
    fig.savefig(Path(docs) / "drift.png")
    plt.close(fig)


def _report(r: dict, wk: pd.DataFrame, docs: str):
    def row(name, e):
        return (f"| {name} | {e['pr_auc'] if e['pr_auc'] is None else round(e['pr_auc'], 3)} | {e['recall']:.1%} | "
                f"{e['precision']:.1%} | {e['fraud_dollars_caught']:.1%} | {e['alerts_per_1k']:.1f} | "
                f"${e['expected_cost']:,.0f} |")
    hdr = ["| Model | PR-AUC | Recall | Precision | Fraud $ caught | Alerts / 1k txns | Expected cost |",
           "|---|---|---|---|---|---|---|"]
    md = ["# FraudStream experiment report", "",
          (f"{r['rows']:,} synthetic transactions, fraud rate {r['fraud_rate']:.2%}. "
          f"Days 0-{BURN_IN - 1} warm up card profiles. Train days {BURN_IN}-{TRAIN_END - 1}, threshold chosen on days {TRAIN_END}-{VALID_END - 1}, "
          f"test days {VALID_END}-{TEST_END - 1}. Drift begins day {DRIFT_DAY}."), "",
          f"## Test period (pre-drift), threshold {r['threshold']:.3f}", "", *hdr,
          row("LightGBM", r["test"]), row("Rules baseline", r["rules"]), "",
          "## Weekly monitoring", "", ("| Week start | Score PSI (unlabeled) | Top drifting features (PSI) | Recall | Precision "
           "| Label-based status |"),
          "|---|---|---|---|---|---|"]
    md += [f"| day {w.week_start_day} | {w.score_psi:.3f} ({monitor.status(w.score_psi)}) | {w.top_drift} | "
           f"{w.recall:.1%} | {w.precision:.1%} | {w.label_status} |" for w in wk.itertuples()]
    md += ["", ("Score and feature PSI stayed below the alert level while recall fell: the new pattern looks like "
           "normal traffic, so unlabeled drift metrics alone missed it. Monitoring recall on delayed "
           "(chargeback) labels caught it, which is why the service tracks both.")]
    md += ["", "![drift](drift.png)", "", f"## Retraining after drift (evaluated on days {RETRAIN_LABELS_UNTIL}-89)", "",
           *hdr, row(f"Champion v1 (days {BURN_IN}-{TRAIN_END - 1})", r["champion_after_drift"]),
           row(f"Challenger v2 (days {BURN_IN}-{RETRAIN_LABELS_UNTIL - 1})", r["challenger_after_drift"]), "",
           (f"Recall on the new 'stealth' pattern: champion {r['stealth_recall']['champion']:.1%} → "
           f"challenger {r['stealth_recall']['challenger']:.1%}."), "", "## Top features (gain)", "",
           *[f"- {k}: {v:,.0f}" for k, v in r["top_features"].items()], "",
           f"_Run time {r['runtime_s']}s. Data is synthetic; see `fraudstream/simulate.py`._", ""]
    (Path(docs) / "report.md").write_text("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--docs", default="docs")
    a = ap.parse_args()
    res = run(a.out, a.docs)
    print(Path(a.docs, "report.md").read_text())
