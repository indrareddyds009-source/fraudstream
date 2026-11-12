import datetime as dt

from fraudstream.features import FEATURES, FeatureStore, build_feature_frame
from fraudstream.simulate import simulate


def test_simulation_is_deterministic_and_ordered():
    a, b = simulate(n_cards=100, days=20, seed=1), simulate(n_cards=100, days=20, seed=1)
    assert a.equals(b)
    assert a["ts"].is_monotonic_increasing
    assert 0.001 < a["is_fraud"].mean() < 0.05


def _t(i, ts, amount, merchant="M1", device="D1", lat=37.77, lon=-122.42, online=True):
    return {"txn_id": f"t{i}", "card_id": "C1", "ts": ts, "amount": amount, "merchant": merchant,
            "category": "online_retail", "online": online, "lat": lat, "lon": lon, "device_id": device,
            "home_lat": 37.77, "home_lon": -122.42}


def test_features_use_only_prior_information():
    fs = FeatureStore()
    t0 = dt.datetime(2026, 1, 1, 12)
    first = fs.process(_t(0, t0, 50))
    assert first["txn_24h"] == 0 and first["new_merchant"] == 0 and first["new_device"] == 0
    for i in range(1, 6):
        fs.process(_t(i, t0 + dt.timedelta(minutes=i), 2.0))
    f = fs.compute(_t(9, t0 + dt.timedelta(minutes=7), 900, merchant="M2", device="EVIL"))
    assert f["small_10m"] == 5 and f["txn_1h"] == 6
    assert f["new_merchant"] == 1 and f["new_device"] == 1 and f["amount_z"] > 3
    # compute() must not mutate state
    assert fs.compute(_t(9, t0 + dt.timedelta(minutes=7), 900, merchant="M2", device="EVIL")) == f


def test_impossible_travel_speed():
    fs = FeatureStore()
    t0 = dt.datetime(2026, 1, 1, 12)
    fs.process(_t(0, t0, 20))
    f = fs.compute(_t(1, t0 + dt.timedelta(minutes=30), 20, lat=40.71, lon=-74.01))  # SF → NYC in 30 min
    assert f["speed_kmh"] > 5000 * 0.99 and f["dist_home_km"] > 4000


def test_feature_frame_shape():
    df = simulate(n_cards=50, days=10, seed=2)
    X, _store = build_feature_frame(df)
    assert list(X.columns) == FEATURES and len(X) == len(df) and not X.isna().any().any()
