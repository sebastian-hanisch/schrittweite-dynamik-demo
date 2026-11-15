"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import sdy_evaluation as ev


def test_formatting_helpers():
    assert ev.de(3.14159, 2) == "3,14" and ev.de(float("inf")) == "∞" and ev.percent(1, 4) == "25,0 %" and ev.percent(0, 0) == "–"


def _step(status="converged", factor=0.9559, c=-0.7, **kw):
    base = {"x": 3.0, "y": 0.1, "eta": 0.3, "c": c, "factor": factor, "D": 8.99, "D1": 8.5937, "status": status, "limit": 6.667, "xe": 2.0, "ye": 0.5, "sharp": 4.25, "steps": 31, "period": None, "D_end": 3.75}
    base.update(kw)
    return base


def test_step_verdict_converged_shrinking_names_factor_steps_and_sharpness():
    kind, text = ev.verdict_step(_step())
    assert kind == "success" and "schrumpft in diesem Schritt um den Faktor 1 − η²c² = 0,9559" in text and "nach 31 Schritten" in text
    assert "S = √(4 + D²) = 4,2500" in text and "2/η = 6,6670" in text


def test_step_verdict_sign_change_and_unchanged_difference():
    assert "wechselt in diesem Schritt das Vorzeichen" in ev.verdict_step(_step(factor=-0.3))[1]
    assert "bleibt in diesem Schritt gleich" in ev.verdict_step(_step(factor=1.0, c=-0.5))[1]


def test_step_verdict_diverged_bounded_and_on_the_curve():
    kind, text = ev.verdict_step(_step(status="diverged", eta=1.2))
    assert kind == "warning" and "Die Bahn divergiert (η = 1,20" in text
    kind, text = ev.verdict_step(_step(status="bounded", period=2))
    assert kind == "info" and "Zyklus der Periode 2" in text
    assert "bis zu Periode 128" in ev.verdict_step(_step(status="bounded", period=0))[1]
    assert "grenzstabil" not in ev.verdict_step(_step(status="bounded", period=0, eta=1.5))[1]
    assert "bei η = 1 ist das Minimum nur grenzstabil" in ev.verdict_step(_step(status="bounded", period=0, eta=1.0))[1]
    kind, text = ev.verdict_step(_step(c=0.0, x=2.0, y=0.5))
    assert kind == "info" and "auf der Kurve x·y = 1" in text and "S = 4,250" in text


def _cl(eta, conv, div, bnd, smax=float("nan"), too=0):
    return {"n": 900, "eta": eta, "converged": conv, "diverged": div, "bounded": bnd, "sharp_max": smax, "limit": 2 / eta, "too_sharp": too}


def test_sharp_verdict_below_threshold_names_counts_and_the_largest_sharpness():
    kind, text = ev.verdict_sharp(_cl(0.8, 485, 415, 0, 2.4041))
    assert kind == "success" and "Von 900 Starts konvergieren 485, 415 divergieren, 0 bleiben" in text
    assert "2,4041" in text and "2/η = 2,5000" in text and "kein erreichtes Minimum hat S ≥ 2/η" in text


def test_sharp_verdict_from_eta_one_and_the_contradiction_case():
    kind, text = ev.verdict_sharp(_cl(1.01, 0, 542, 358))
    assert kind == "success" and "kein stabiles Minimum, also konvergiert kein Start" in text and "2/η = 1,9802" in text
    kind, text = ev.verdict_sharp(_cl(1.0, 0, 534, 366))                       # η = 1: grenzstabil, nicht „kein stabiles Minimum“
    assert kind == "info" and "nur grenzstabil" in text and "kein stabiles Minimum" not in text and "konvergiert kein Start" not in text
    assert "keine Zyklen" in text and "366 bleiben beschränkt" in text
    kind, text = ev.verdict_sharp(_cl(1.0, 0, 900, 0))
    assert kind == "info" and "nur grenzstabil" in text and "keine Zyklen" not in text
    kind, text = ev.verdict_sharp(_cl(0.5, 0, 900, 0))
    assert kind == "info" and "konvergiert kein Start" not in text
    kind, text = ev.verdict_sharp(_cl(0.8, 400, 500, 0, 2.6, too=3))
    assert kind == "warning" and "3 der erreichten Minima haben Schärfe S ≥ 2/η" in text


def test_diag_verdict_all_branches():
    th = [1.0, 1.2361, 1.2880, 1.2992]
    kind, text = ev.verdict_diag({"eta": 1.1, "period": 2, "lyapunov": -0.05}, th, 1.3137)
    assert kind == "success" and "Zyklus der Periode 2" in text and "kein Chaos" in text
    assert "Verdopplungsschwellen bei 1, 1,2361, 1,2880, 1,2992; Chaos auf dem Gitter ab η = 1,3137." in text
    kind, text = ev.verdict_diag({"eta": 1.5, "period": 0, "lyapunov": 0.49}, th, 1.3137)
    assert kind == "success" and "+0,490" in text and "chaotisch" in text
    kind, text = ev.verdict_diag({"eta": 1.0, "period": 0, "lyapunov": 0.003}, th, None)
    assert kind == "info" and "nicht als Chaos eingestuft" in text and text.endswith("1,2992.") and "nur grenzstabil" in text
    assert "grenzstabil" not in ev.verdict_diag({"eta": 1.1, "period": 0, "lyapunov": 0.001}, th, None)[1]
    assert ev.verdict_diag({"eta": 2.0, "period": None, "lyapunov": float("nan")}, th, None)[0] == "warning"


def test_efs_verdict_without_and_with_bounded_orbits():
    kind, text = ev.verdict_efs({"n_bounded": 0, "eta": 0.5, "converged": 767, "diverged": 133})
    assert kind == "info" and "767 konvergieren, 133 divergieren" in text
    r = {"n_bounded": 358, "eta": 1.01, "periods": {2: 358}, "exact": True, "n_chaotic": 0, "lyapunov_median": -0.01, "mean_median": 2.005, "range_min": 1.705, "range_max": 2.305, "above_median": 0.5}
    kind, text = ev.verdict_efs(r)
    assert "Periode 2 (358 Bahnen)" in text and "exaktes Zeitmittel über eine Periode" in text and "2,005" in text and "1,705 bis 2,305" in text and "0,500" in text
    assert "ob Netze sich so verhalten, ist hier nicht gemessen" in text and "Lyapunov" not in text
    r.update(periods={0: 185}, exact=False, n_chaotic=185, lyapunov_median=0.49, n_bounded=185)
    text = ev.verdict_efs(r)[1]
    assert "ohne Periode bis 128 (185 Bahnen)" in text and "Mittel über 4000 Schritte" in text and "positiv (über 0,003): 185 von 185" in text
    assert "grenzstabil" not in text and "weder Zyklus noch Chaos" not in text
    r.update(eta=1.0, n_chaotic=0, n_bounded=366, periods={0: 366})
    assert "weder Zyklus noch Chaos" in ev.verdict_efs(r)[1]


def test_newton_verdict_with_and_without_the_frozen_dimension():
    rep = {"shares": [0.3339, 0.3338, 0.3324], "radius": 1.5, "n": 1_000_000}
    kind, text = ev.verdict_newton(rep, None)
    assert "33,4 %, 33,4 %, 33,2 %" in text and "1.000.000 Zufallspunkte" in text and "liegt nicht vor" in text
    dim = {"unsicherheit": {"dimension": 1.4466}, "box": {"dimension_4_bis_64": 1.4538, "dimension_2_bis_16": 1.4770}}
    text = ev.verdict_newton(rep, dim)[1]
    assert "1,45 bis 1,48" in text and "Unsicherheitsexponent 1,447" in text and "Box-Zählung 1,454 und 1,477" in text and "keine Literaturgegenprobe" in text
