"""Synthetic card-transaction stream with realistic legitimate behaviour and labeled fraud.

Legit: each card has a home city, spending level, preferred merchants, a device, waking-hours
activity, and occasional genuine travel (so "far from home" alone is not fraud).
Fraud patterns (days 0-90):  card_testing, account_takeover, skimming.
Concept drift (from DRIFT_DAY): a new "stealth" pattern on the victim's own device near home
(gift cards, mid amounts), plus ~15% price inflation in legit spend: the kind of shift
a production model must detect and adapt to.
"""
from __future__ import annotations

import datetime as dt
import math
import random

import pandas as pd

CITIES = {"San Francisco": (37.77, -122.42), "Los Angeles": (34.05, -118.24), "Seattle": (47.61, -122.33),
          "Dallas": (32.78, -96.80), "Chicago": (41.88, -87.63), "New York": (40.71, -74.01),
          "Miami": (25.76, -80.19), "Denver": (39.74, -104.99), "Atlanta": (33.75, -84.39),
          "Boston": (42.36, -71.06)}
CATEGORIES = {  # category: (amount multiplier, online share)
    "grocery": (1.2, 0.05), "restaurant": (0.8, 0.02), "gas": (1.1, 0.0), "online_retail": (1.5, 1.0),
    "electronics": (4.0, 0.5), "travel": (6.0, 0.7), "gift_cards": (2.0, 0.6), "pharmacy": (0.7, 0.1),
    "entertainment": (0.9, 0.4)}
LEGIT_CAT_W = [22, 20, 12, 18, 3, 2, 1, 10, 12]
START = dt.datetime(2026, 1, 1)
DAYS = 90
DRIFT_DAY = 60


def haversine_km(a, b) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def _jitter(r, city):
    lat, lon = CITIES[city]
    return lat + r.gauss(0, 0.05), lon + r.gauss(0, 0.05)


def simulate(n_cards: int = 2000, days: int = DAYS, seed: int = 11, fraud_events_per_day: float = 4.0,
             drift_day: int = DRIFT_DAY) -> pd.DataFrame:
    r = random.Random(seed)
    cats = list(CATEGORIES)
    merchants = {c: [f"{c[:4].upper()}-{i:03d}" for i in range(60)] for c in cats}
    cards = []
    for i in range(n_cards):
        home = r.choice(list(CITIES))
        cards.append({"card_id": f"C{i:05d}", "home": home, "home_ll": CITIES[home],
                      "level": math.exp(r.gauss(3.4, 0.5)), "rate": r.uniform(0.4, 2.5),
                      "device": f"D{i:05d}a", "fav": {c: r.sample(merchants[c], 6) for c in cats},
                      "trip": (r.randint(0, days - 6), r.choice(list(CITIES))) if r.random() < 0.25 else None,
                      "micro": r.random() < 0.12,       # app-store / transit micro-payments in bursts
                      "night_owl": r.random() < 0.10})  # legitimately active overnight
    rows = []

    def add(card, ts, cat, amount, merchant, city, device, online, fraud, ftype):
        lat, lon = _jitter(r, city)
        rows.append({"ts": ts, "card_id": card["card_id"], "merchant": merchant, "category": cat,
                     "amount": round(amount, 2), "online": int(online), "lat": round(lat, 4), "lon": round(lon, 4),
                     "device_id": device if online else None, "home_lat": card["home_ll"][0],
                     "home_lon": card["home_ll"][1], "is_fraud": int(fraud), "fraud_type": ftype})

    # legitimate activity
    for card in cards:
        for d in range(days):
            infl = 1.15 if d >= drift_day else 1.0
            for _ in range(_poisson(r, card["rate"])):
                cat = r.choices(cats, LEGIT_CAT_W)[0]
                mult, online_share = CATEGORIES[cat]
                city = card["home"]
                if card["trip"] and card["trip"][0] <= d < card["trip"][0] + 5:
                    city = card["trip"][1]
                if card["night_owl"] and r.random() < 0.35:
                    hour = r.choice([22, 23, 0, 1, 2, 3, 4])
                else:
                    hour = r.randint(0, 5) if r.random() < 0.03 else min(23, max(6, int(r.gauss(15, 4))))
                ts = START + dt.timedelta(days=d, hours=hour, minutes=r.randint(0, 59), seconds=r.randint(0, 59))
                amt = card["level"] * mult * math.exp(r.gauss(0, 0.45)) * infl
                m = r.choice(card["fav"][cat]) if r.random() < 0.8 else r.choice(merchants[cat])
                if r.random() < 0.003:  # customer gets a new phone/laptop: it becomes their device from now on
                    card["device"] = f"{card['card_id']}-{r.randint(0, 10**6):06d}"
                dev = card["device"]
                add(card, ts, cat, amt, m, city, dev, r.random() < online_share, False, None)
            if card["micro"] and r.random() < 0.3:
                t0 = START + dt.timedelta(days=d, hours=r.randint(7, 22), minutes=r.randint(0, 50))
                m = r.choice(merchants["entertainment"])
                for k in range(r.randint(2, 6)):
                    add(card, t0 + dt.timedelta(seconds=45 * k + r.randint(0, 60)), "entertainment",
                        r.uniform(0.99, 4.99), m, card["home"], card["device"], True, False, None)

    # fraud events
    n_events = max(1, int(days * fraud_events_per_day * n_cards / 2000))
    for _ in range(n_events):
        d = r.randrange(days)
        card = r.choice(cards)
        base = START + dt.timedelta(days=d, hours=r.randint(0, 23), minutes=r.randint(0, 59))
        far_city = r.choice([c for c in CITIES if c != card["home"]])
        near = card["home"]  # fraud via VPN / local mule: geography gives no signal
        kind = r.choice(["card_testing", "account_takeover", "skimming"])
        if d >= drift_day and r.random() < 0.6:
            kind = "stealth"
        if kind == "card_testing":
            m = r.choice(merchants["online_retail"])
            dev = f"X{r.randint(0, 10**6):06d}"
            far_city = far_city if r.random() < 0.5 else near
            for k in range(r.randint(4, 12)):
                add(card, base + dt.timedelta(seconds=40 * k + r.randint(0, 30)), "online_retail",
                    r.uniform(0.5, 4.0), m, far_city, dev, True, True, kind)
            add(card, base + dt.timedelta(minutes=12), "electronics", r.uniform(300, 1500),
                r.choice(merchants["electronics"]), far_city, dev, True, True, kind)
        elif kind == "account_takeover":
            dev = f"X{r.randint(0, 10**6):06d}"
            far_city = far_city if r.random() < 0.5 else near
            night = base.replace(hour=r.choice([0, 1, 2, 3, 4]) if r.random() < 0.5 else r.randint(9, 22))
            for k in range(r.randint(1, 3)):
                cat = r.choice(["electronics", "gift_cards", "travel"])
                add(card, night + dt.timedelta(minutes=7 * k), cat, card["level"] * r.uniform(3, 20),
                    r.choice(merchants[cat]), far_city, dev, True, True, kind)
        elif kind == "skimming":
            far_city = far_city if r.random() < 0.6 else near
            for k in range(r.randint(2, 5)):
                cat = r.choice(["gas", "electronics", "grocery"])
                add(card, base + dt.timedelta(minutes=25 * k), cat, card["level"] * r.uniform(3, 9),
                    r.choice(merchants[cat]), far_city, None, False, True, kind)
        else:  # stealth: victim's own device, home city, waking hours, gift cards at mid amounts
            t = base.replace(hour=r.randint(10, 20))
            for k in range(r.randint(2, 4)):
                add(card, t + dt.timedelta(minutes=r.randint(20, 90) * k), "gift_cards",
                    card["level"] * r.uniform(2.5, 5) * 1.15, r.choice(merchants["gift_cards"]), card["home"],
                    card["device"], True, True, kind)

    df = pd.DataFrame(rows).sort_values("ts", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])
    return df


def _poisson(r, lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= r.random()
        if p <= L:
            return k
        k += 1


if __name__ == "__main__":
    df = simulate()
    print(len(df), df.is_fraud.mean(), df.groupby("fraud_type").size().to_dict())
