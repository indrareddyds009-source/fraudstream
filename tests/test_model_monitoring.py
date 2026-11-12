import numpy as np

from fraudstream import explain, model, monitor


def test_model_beats_rules_and_is_saved(trained):
    out, docs, r = trained
    assert r["test"]["pr_auc"] > r["rules"]["pr_auc"] + 0.2
    assert r["test"]["expected_cost"] < r["rules"]["expected_cost"]
    assert (out / "model.txt").exists() and (out / "v2" / "model.txt").exists()
    assert (docs / "report.md").exists() and (docs / "drift.png").exists()


def test_retraining_recovers_after_drift(trained):
    _, _, r = trained
    assert r["challenger_after_drift"]["recall"] > r["champion_after_drift"]["recall"]
    assert r["stealth_recall"]["challenger"] > r["stealth_recall"]["champion"]


def test_threshold_minimizes_cost():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    s = np.clip(y * 0.6 + rng.random(5000) * 0.5, 0, 1)
    amt = rng.uniform(10, 500, 5000)
    thr = model.choose_threshold(s, y, amt)
    c = model.evaluate(s, y, amt, thr)["expected_cost"]
    for other in (0.2, 0.5, 0.9):
        assert c <= model.evaluate(s, y, amt, other)["expected_cost"] + 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = monitor.reference_bins(rng.normal(0, 1, 20000))
    assert monitor.psi(ref, rng.normal(0, 1, 5000)) < monitor.PSI_WARN
    assert monitor.psi(ref, rng.normal(1.0, 1, 5000)) > monitor.PSI_ALERT
    binary = monitor.reference_bins(np.array([0] * 900 + [1] * 100))
    assert monitor.psi(binary, np.array([0] * 500 + [1] * 500)) > monitor.PSI_ALERT
    assert monitor.performance_status(0.95, 0.70) == "ALERT" and monitor.performance_status(0.95, 0.9) == "ok"


def test_grounding_check():
    facts = "Risk score: 0.97\n- distance from home: 2,100 km"
    assert explain.is_grounded("Score 0.97; the purchase was 2,100 km from home.", facts)
    assert not explain.is_grounded("Score 0.97; the customer lost $5,000.", facts)
