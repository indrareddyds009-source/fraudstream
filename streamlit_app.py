"""FraudStream live demo (Streamlit). Run locally: streamlit run streamlit_app.py"""
import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fraudstream import pipeline
from fraudstream.scorer import FraudScorer
from fraudstream.simulate import CATEGORIES, CITIES, START, simulate

st.set_page_config(page_title="FraudStream", page_icon="🛡️", layout="wide")
ART, DOCS, N_CARDS, SEED = "artifacts", "demo_docs", 1000, 11


@st.cache_resource(show_spinner="First load: simulating 90 days of card transactions and training the model (~30 s)…")
def setup():
    results = pipeline.run(ART, DOCS, n_cards=N_CARDS, seed=SEED)
    df = simulate(n_cards=N_CARDS, seed=SEED)
    df["day"] = (df["ts"] - START).dt.days
    base = FraudScorer(ART)
    base.warm(df[df["day"] < 50].to_dict("records"))
    return results, df, base


results, df, base_scorer = setup()
if "scorer" not in st.session_state:
    st.session_state.scorer = copy.deepcopy(base_scorer)
    st.session_state.n = 0
sc: FraudScorer = st.session_state.scorer

st.title("🛡️ FraudStream")
st.caption("Real-time card-fraud scoring with an online feature store, cost-based thresholds, drift monitoring and "
           "explainable reason codes. All data is synthetic. "
           "[Source code](https://github.com/indrareddyds009-source/fraudstream)")

t = results["test"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC (test)", f"{t['pr_auc']:.3f}", f"{t['pr_auc'] - results['rules']['pr_auc']:+.2f} vs rules")
c2.metric("Recall", f"{t['recall']:.0%}")
c3.metric("Fraud $ caught", f"{t['fraud_dollars_caught']:.0%}")
c4.metric("Alerts / 1k txns", f"{t['alerts_per_1k']:.1f}")

tab1, tab2, tab3 = st.tabs(["🔎 Score a transaction", "⚡ Live stream replay", "📈 Model & drift report"])

with tab1:
    hist = df[df["day"] < 50]
    cards = hist.groupby("card_id").size()
    cards = cards[cards > 40].index[:200]
    left, right = st.columns([1, 1])
    with left:
        card = st.selectbox("Card", cards, help="A synthetic customer with 50 days of history already in the feature store")
        prof = hist[hist["card_id"] == card]
        home_city = min(CITIES, key=lambda c: (CITIES[c][0] - prof["home_lat"].iloc[0]) ** 2 + (CITIES[c][1] - prof["home_lon"].iloc[0]) ** 2)
        st.caption(f"Home: **{home_city}** · typical spend **${prof['amount'].median():,.2f}** · "
                   f"{len(prof)} past transactions")
        scenario = st.radio("Quick scenario", ["Normal purchase", "Account takeover", "Card testing burst", "Custom"], horizontal=True)
        presets = {
            "Normal purchase": dict(amount=float(round(prof["amount"].median(), 2)), category="grocery", online=False,
                                    city=home_city, new_device=False, hour=14),
            "Account takeover": dict(amount=1450.0, category="electronics", online=True,
                                     city=[c for c in CITIES if c != home_city][0], new_device=True, hour=2),
            "Card testing burst": dict(amount=2.5, category="online_retail", online=True, city=home_city, new_device=True, hour=21),
            "Custom": dict(amount=60.0, category="restaurant", online=False, city=home_city, new_device=False, hour=19),
        }[scenario]
        amount = st.number_input("Amount ($)", 0.5, 20000.0, presets["amount"], step=10.0)
        category = st.selectbox("Category", list(CATEGORIES), index=list(CATEGORIES).index(presets["category"]))
        city = st.selectbox("Location", list(CITIES), index=list(CITIES).index(presets["city"]))
        online = st.checkbox("Card-not-present (online)", presets["online"])
        new_device = st.checkbox("Unrecognized device", presets["new_device"])
        hour = st.slider("Hour of day", 0, 23, presets["hour"])
        burst = st.number_input("Send this many times in a row (1 min apart)", 1, 12, 6 if scenario == "Card testing burst" else 1)
        go = st.button("Score transaction", type="primary")
    with right:
        if go:
            last = prof.iloc[-1]
            dev = f"DEMO-NEW-{st.session_state.n}" if new_device else (last["device_id"] or f"{card}-dev")
            base_ts = START + dt.timedelta(days=50, hours=hour)
            out = None
            for k in range(int(burst)):
                st.session_state.n += 1
                txn = {"txn_id": f"DEMO{st.session_state.n:06d}", "ts": base_ts + dt.timedelta(minutes=k), "card_id": card,
                       "merchant": f"DEMO-{category[:4].upper()}", "category": category, "amount": amount,
                       "online": int(online), "lat": CITIES[city][0], "lon": CITIES[city][1],
                       "device_id": dev if online else None, "home_lat": last["home_lat"], "home_lon": last["home_lon"]}
                out = sc.score(txn)
            color = {"approve": "green", "review": "orange", "decline": "red"}[out["decision"]]
            st.markdown(f"### Decision: :{color}[{out['decision'].upper()}]")
            st.metric("Fraud score", f"{out['score']:.3f}", help=f"Review threshold {sc.card.threshold:.3f} (chosen to minimize expected $ cost)")
            st.caption(f"Scored in {out['latency_ms']:.2f} ms including feature computation")
            exp = sc.explain(out["txn_id"])
            if exp["reasons"]:
                st.markdown("**Why (TreeSHAP reason codes):**")
                st.dataframe(pd.DataFrame(exp["reasons"])[["label", "value", "contribution"]], hide_index=True, width="stretch")
            st.info(exp["note"])
        else:
            st.markdown("Pick a scenario and press **Score transaction**. Try *Account takeover* or *Card testing burst*.")
        if st.button("Reset feature store"):
            st.session_state.scorer = copy.deepcopy(base_scorer)
            st.rerun()

with tab2:
    st.markdown("Replays unseen transactions (days 50–59) through the live scorer, exactly as the API would see them.")
    n = st.select_slider("Transactions to replay", [500, 1000, 2000, 5000], value=2000)
    if st.button("Run replay", type="primary"):
        rs = copy.deepcopy(base_scorer)
        stream = df[(df["day"] >= 50) & (df["day"] < 60)].head(n)
        rows, lat = [], []
        prog = st.progress(0.0)
        for i, txn in enumerate(stream.to_dict("records")):
            r = rs.score(txn)
            lat.append(r["latency_ms"])
            if r["decision"] != "approve":
                rows.append({"time": txn["ts"], "card": txn["card_id"], "amount": txn["amount"], "category": txn["category"],
                             "decision": r["decision"], "score": r["score"], "actually fraud": bool(txn["is_fraud"]),
                             "top reason": f"{r['reasons'][0]['label']}: {r['reasons'][0]['value']}" if r["reasons"] else ""})
            if i % 100 == 0:
                prog.progress(i / len(stream))
        prog.empty()
        fl = pd.DataFrame(rows)
        n_fraud = int(stream["is_fraud"].sum())
        caught = int(fl["actually fraud"].sum()) if len(fl) else 0
        a, b, c, d = st.columns(4)
        a.metric("Transactions", f"{len(stream):,}")
        b.metric("Flagged", len(fl))
        c.metric("Fraud caught", f"{caught}/{n_fraud}")
        d.metric("Latency p50 / p99", f"{np.percentile(lat, 50):.2f} / {np.percentile(lat, 99):.2f} ms")
        st.dataframe(fl, hide_index=True, width="stretch")

with tab3:
    st.markdown("#### Pre-drift test period vs. hand-written rules")
    cols = ["pr_auc", "recall", "precision", "fraud_dollars_caught", "alerts_per_1k", "expected_cost"]
    st.dataframe(pd.DataFrame({"LightGBM": results["test"], "Rules baseline": results["rules"]}).T[cols].round(3), width="stretch")
    st.markdown("#### Weekly monitoring: drift starts on day 60")
    wk = pd.DataFrame(results["weekly"])
    st.dataframe(wk.round(3), hide_index=True, width="stretch")
    img = Path(DOCS) / "drift.png"
    if img.exists():
        st.image(str(img))
    st.markdown("#### Retraining after drift (days 75–89)")
    st.dataframe(pd.DataFrame({"Champion v1": results["champion_after_drift"], "Challenger v2": results["challenger_after_drift"]}).T[cols].round(3),
                 width="stretch")
    s = results["stealth_recall"]
    st.success(f"Recall on the new 'stealth' fraud pattern: {s['champion']:.0%} → {s['challenger']:.0%} after retraining.")
