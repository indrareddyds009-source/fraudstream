import datetime as dt

from fastapi.testclient import TestClient

from fraudstream.scorer import FraudScorer


def _txn(i, minutes, amount, device="D1", merchant="M1", lat=37.77, lon=-122.42):
    return {"txn_id": f"x{i}", "ts": (dt.datetime(2026, 2, 1, 14) + dt.timedelta(minutes=minutes)).isoformat(),
            "card_id": "CTEST", "merchant": merchant, "category": "online_retail", "amount": amount, "online": True,
            "lat": lat, "lon": lon, "device_id": device, "home_lat": 37.77, "home_lon": -122.42}


def test_api_scores_explains_and_monitors(trained, monkeypatch):
    out, _, _ = trained
    monkeypatch.setenv("FRAUDSTREAM_ARTIFACTS", str(out))
    from fraudstream import api
    api.scorer.cache_clear()
    c = TestClient(api.app)
    assert c.get("/health").json()["model_version"] == "v1"
    # normal history, then a card-testing burst from a new device far away, then a big purchase
    for i in range(20):
        r = c.post("/score", json=_txn(i, i * 1440, 40 + i % 7))
        assert r.status_code == 200
    for k in range(8):
        c.post("/score", json=_txn(100 + k, 30000 + k, 1.5, device="EVIL", merchant="SHADY", lat=40.71, lon=-74.01))
    hit = c.post("/score", json=_txn(200, 30010, 1200, device="EVIL", merchant="ELEC", lat=40.71, lon=-74.01)).json()
    assert hit["decision"] in ("review", "decline") and hit["reasons"]
    e = c.get("/explain/x200").json()
    assert e["note"].startswith("Risk score") and len(e["reasons"]) >= 1
    assert c.get("/explain/nope").status_code == 404
    assert c.get("/monitor/drift").json()["status"] == "insufficient_data"
    assert c.post("/score", json={**_txn(300, 1, 10), "amount": -5}).status_code == 422


def test_scorer_latency_budget(trained):
    out, _, _ = trained
    sc = FraudScorer(str(out))
    t = _txn(1, 0, 25)
    t["ts"] = dt.datetime.fromisoformat(t["ts"])
    assert sc.score(t)["latency_ms"] < 50
