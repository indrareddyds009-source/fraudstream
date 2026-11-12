"""FraudStream live demo. Run: streamlit run app/streamlit_app.py"""
import datetime as dt
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make the fraudstream package importable on Streamlit Cloud

from fraudstream import pipeline  # noqa: E402
from fraudstream.scorer import FraudScorer  # noqa: E402
from fraudstream.simulate import CITIES, START, simulate  # noqa: E402

st.set_page_config(page_title="FraudStream demo", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Simulating a transaction stream and training the model (about 20 seconds, once)…")
def setup():
    out = tempfile.mkdtemp()
    results = pipeline.run(out, tempfile.mkdtemp(), n_cards=800, seed=11)
    df = simulate(n_cards=800, seed=11)
    return out, results, df


ART, RESULTS, DF = setup()
DAY = (DF["ts"] - START).dt.days


def fresh_scorer(history_until_day=50):
    sc = FraudScorer(ART)
    sc.warm(DF[DAY < history_until_day].to_dict("records"))
    return sc


st.title("🛡️ FraudStream: real-time card fraud scoring")
st.caption("Online feature store → LightGBM → cost-optimal decision → SHAP reason codes · drift monitoring · "
           "synthetic data only · code: github.com/indrareddyds009-source/fraudstream")

tab1, tab2, tab3 = st.tabs(["Try a transaction", "Live stream replay", "Model & drift report"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.write("Pick a card and a scenario. The card's real history is loaded into the feature store first, "
             "then your transaction(s) are scored exactly as the API would.")
    cards = DF[DAY < 50].groupby("card_id").size()
    cards = cards[cards > 40].index[:40].tolist()
    c1, c2 = st.columns([1, 2])
    card = c1.selectbox("Card", cards)
    hist = DF[(DF.card_id == card) & (DAY < 50)]
    home = (hist.home_lat.iloc[0], hist.home_lon.iloc[0])
    home_city = min(CITIES, key=lambda c: (CITIES[c][0] - home[0]) ** 2 + (CITIES[c][1] - home[1]) ** 2)
    c1.caption(f"Home: {home_city} · {len(hist)} past transactions · typical spend ${hist.amount.median():,.2f}")
    scenario = c2.radio("Scenario", [
        "Normal: grocery purchase near home",
        "Card testing: 8 tiny online charges in 6 minutes, then a $900 electronics order",
        "Account takeover: new device, another city, 3 a.m., $1,400 gift cards",
        "Custom transaction",
    ])
    far_city = next(c for c in CITIES if c != home_city)
    base = dt.datetime(2026, 2, 20, 14, 0)
    dev = hist.device_id.dropna().iloc[-1] if hist.device_id.notna().any() else None

    def txn(i, minutes, amount, cat, merchant, city, online, device):
        lat, lon = CITIES[city]
        return {"txn_id": f"demo-{i}", "ts": base + dt.timedelta(minutes=minutes), "card_id": card, "merchant": merchant,
                "category": cat, "amount": amount, "online": online, "lat": lat, "lon": lon,
                "device_id": device if online else None, "home_lat": home[0], "home_lon": home[1]}

    if scenario.startswith("Normal"):
        txns = [txn(1, 0, round(hist.amount.median() * 1.2, 2), "grocery", "GROC-001", home_city, False, None)]
    elif scenario.startswith("Card testing"):
        txns = [txn(i, i * 0.7, round(1 + i * 0.37, 2), "online_retail", "ONLI-999", far_city, True, "X-unknown")
                for i in range(8)]
        txns.append(txn(99, 6, 900.0, "electronics", "ELEC-777", far_city, True, "X-unknown"))
    elif scenario.startswith("Account takeover"):
        txns = [txn(1, 0, 1400.0, "gift_cards", "GIFT-042", far_city, True, "X-new-laptop")]
        txns[0]["ts"] = base.replace(hour=3)
    else:
        a, b, c = st.columns(3)
        amount = a.number_input("Amount ($)", 1.0, 20000.0, 250.0)
        cat = b.selectbox("Category", ["grocery", "restaurant", "gas", "online_retail", "electronics", "travel",
                                       "gift_cards", "pharmacy", "entertainment"])
        city = c.selectbox("City", list(CITIES), index=list(CITIES).index(home_city))
        d1, d2 = st.columns(2)
        online = d1.checkbox("Online (card-not-present)", value=True)
        new_device = d2.checkbox("Unrecognized device", value=False)
        txns = [txn(1, 0, amount, cat, f"{cat[:4].upper()}-123", city, online, "X-new" if new_device else dev)]

    if st.button("Score", type="primary"):
        sc = fresh_scorer()
        res = [sc.score(t) for t in txns]
        last = res[-1]
        color = {"approve": "green", "review": "orange", "decline": "red"}[last["decision"]]
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk score", f"{last['score']:.3f}")
        m2.markdown(f"### :{color}[{last['decision'].upper()}]")
        m3.metric("Scoring latency", f"{last['latency_ms']:.2f} ms")
        if len(res) > 1:
            st.dataframe(pd.DataFrame([{"#": i + 1, "amount": t["amount"], "score": r["score"], "decision": r["decision"]}
                                       for i, (t, r) in enumerate(zip(txns, res, strict=True))]), hide_index=True)
        exp = sc.explain(last["txn_id"])
        st.subheader("Why")
        st.write(exp["note"])
        if exp["reasons"]:
            st.dataframe(pd.DataFrame(exp["reasons"])[["label", "value", "contribution"]]
                         .rename(columns={"label": "signal", "contribution": "push on score (log-odds)"}), hide_index=True)

# ---------------------------------------------------------------- tab 2
with tab2:
    st.write("Replays real (synthetic) traffic from days 50–59 through the live scorer.")
    n = st.slider("Transactions to replay", 1000, 20000, 5000, step=1000)
    if st.button("Run replay", type="primary"):
        sc = fresh_scorer()
        stream = DF[(DAY >= 50) & (DAY < 60)].head(n)
        lat, rows = [], []
        t0 = time.perf_counter()
        prog = st.progress(0.0)
        for i, t in enumerate(stream.to_dict("records")):
            r = sc.score(t)
            lat.append(r["latency_ms"])
            if r["decision"] != "approve":
                rows.append({"date": t["ts"].strftime("%b %d %H:%M"), "amount": t["amount"], "category": t["category"],
                             "score": r["score"], "decision": r["decision"], "actually fraud": bool(t["is_fraud"])})
            if i % 500 == 0:
                prog.progress(i / len(stream))
        prog.progress(1.0)
        wall = time.perf_counter() - t0
        fraud = int(stream.is_fraud.sum())
        caught = sum(r["actually fraud"] for r in rows)
        k = st.columns(4)
        k[0].metric("Throughput", f"{len(stream) / wall:,.0f} txn/s")
        k[1].metric("p99 latency", f"{np.percentile(lat, 99):.2f} ms")
        k[2].metric("Flagged", f"{len(rows)} / {len(stream):,}")
        k[3].metric("Fraud caught", f"{caught} / {fraud}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 3
with tab3:
    r = RESULTS
    st.write(f"{r['rows']:,} transactions · fraud rate {r['fraud_rate']:.2%} · decision threshold {r['threshold']:.3f} "
             "(chosen to minimize missed-fraud dollars + $5 per review)")
    cols = ("pr_auc", "recall", "precision", "fraud_dollars_caught")
    tbl = pd.DataFrame([{"model": name, **{k: r[key][k] for k in cols}} for name, key in (
        ("LightGBM (pre-drift test)", "test"), ("Hand-written rules", "rules"),
        ("Champion v1 after drift", "champion_after_drift"), ("Challenger v2 (retrained)", "challenger_after_drift"))])
    st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns if c != "model"}), hide_index=True)
    wk = pd.DataFrame(r["weekly"])
    st.subheader("Weekly monitoring")
    st.line_chart(wk.set_index("week_start_day")[["recall", "fraud_dollars_caught"]])
    st.dataframe(wk[["week_start_day", "score_psi", "top_drift", "recall", "precision"]], hide_index=True)
    st.caption("A new fraud pattern starts on day 60. Score PSI stays low because the attack looks normal; "
               "the label-based recall monitor catches it, and retraining recovers it.")
