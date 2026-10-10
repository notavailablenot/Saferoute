import numpy as np

from saferoute.evaluation.metrics import classification_report, detection_report, latency_summary


def test_classification_report_values():
    r = classification_report(["A", "A", "B", "B"], ["A", "B", "B", "B"], ["A", "B", "C"])
    assert r["accuracy"] == 0.75
    assert r["per_class"]["A"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert r["per_class"]["B"]["precision"] == 0.6667 and r["per_class"]["B"]["recall"] == 1.0
    assert abs(r["macro"]["f1"] - 0.7333) < 1e-3  # mean over classes present (A, B)
    assert r["confusion_matrix"][0] == [1, 1, 0]


def test_detection_report_perfect_and_miss():
    g = [np.array([[0, 0, 10, 10]], float), np.array([[20, 20, 40, 40]], float)]
    perfect = [(g[0].copy(), np.array([0.9])), (g[1].copy(), np.array([0.8]))]
    r = detection_report(perfect, g)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["map50"] > 0.99 and r["map50_95"] > 0.99
    one_wrong = [(g[0].copy(), np.array([0.9])), (np.array([[100, 100, 120, 120]], float), np.array([0.8]))]
    r = detection_report(one_wrong, g)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1 and r["f1"] == 0.5


def test_latency_summary():
    s = latency_summary([10.0] * 99 + [20.0])
    assert s["p50_ms"] == 10.0 and s["max_ms"] == 20.0 and s["n"] == 100
