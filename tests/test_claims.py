"""Jede Zahl der README (Abschnitt „Befunde“) wird hier nachgerechnet. Die Rechnung ist IEEE-Arithmetik ohne Zufall (Raster) bzw. mit dem eigenen Generator SplitMix64 und deshalb
auf jeder Plattform dieselbe; Zählungen stehen exakt, Kennzahlen über Summen als Band (± 2 bei Zählungen am Rand, wie im Plan vorgesehen: hier exakt, weil gemessen)."""

import numpy as np
import pytest

import sdy_algorithm as A
import sdy_constants as C
import sdy_core as K
import sdy_evaluation as ev
import sdy_fractal as F


# (η, konvergiert, divergiert, beschränkt ohne Konvergenz, größte Schärfe der erreichten Minima)
GRID_ROWS = [
    (0.2, 900, 0, 0, 6.6060), (0.49, 780, 120, 0, 4.0108), (0.5, 767, 133, 0, 3.9124), (0.51, 764, 136, 0, 3.6771), (0.8, 485, 415, 0, 2.4041),
    (0.95, 393, 507, 0, 2.0844), (0.99, 372, 528, 0, 2.0171), (1.0, 0, 534, 366, None), (1.01, 0, 542, 358, None), (1.05, 0, 558, 342, None),
    (1.3, 0, 654, 246, None), (1.5, 0, 715, 185, None), (1.9, 0, 833, 67, None), (1.99, 0, 877, 23, None), (2.0, 0, 900, 0, None), (2.01, 0, 900, 0, None),
]


@pytest.mark.parametrize("eta, conv, div, bnd, smax", GRID_ROWS)
def test_counts_of_the_900_starts(eta, conv, div, bnd, smax):
    """900 Starts im Quadrat [−2,5; 2,5]² mit Versatz (0,0137; −0,0291), 6000 Schritte: konvergiert / divergiert / beschränkt, größte Schärfe der erreichten Minima, immer unter 2/η."""
    cl = K.classify_grid(eta)
    assert (cl["converged"], cl["diverged"], cl["bounded"]) == (conv, div, bnd)
    if smax is not None:
        assert round(cl["sharp_max"], 4) == smax and cl["sharp_max"] < 2 / eta
    assert cl["too_sharp"] == 0


def test_at_eta_one_the_bounded_orbits_creep_towards_the_minimum():
    """η = 1: (1; 1) hat S = 2 = 2/η und ist nur grenzstabil. Die 366 beschränkten Bahnen sind keine Zyklen: der Verlust liegt nach 6000 Schritten bei allen unter 5·10⁻⁵
    und fällt weiter (nach 20 000 Schritten unter 1,4·10⁻⁵). Die Verdikte dürfen hier nicht „kein stabiles Minimum, also konvergiert kein Start“ sagen
    (Fund der unabhängigen Prüfung)."""
    cl = K.classify_grid(1.0)
    assert cl["bounded"] == 366 and cl["converged"] == 0
    x, y = cl["x"][cl["bnd"]], cl["y"][cl["bnd"]]
    l6000 = K.loss(x, y)
    assert l6000.max() < 5e-5 and (K.sharpness(x, y) > 1.9).all()
    for _ in range(14_000):
        x, y = K.gd_step(x, y, 1.0)
    l20000 = K.loss(x, y)
    assert l20000.max() < 1.4e-5 and (l20000 < l6000).all()
    text = ev.verdict_sharp(cl)[1]
    assert "kein stabiles Minimum, also konvergiert kein Start" not in text and "nur grenzstabil" in text and "langsam gegen das Minimum" in text
    assert "kein stabiles Minimum, also konvergiert kein Start" in ev.verdict_sharp(K.classify_grid(1.01))[1]


def test_no_reached_minimum_is_sharper_than_2_over_eta_in_nine_step_sizes():
    """Zählung über die 9 Schrittweiten 0,49, 0,51, 0,99, 1,01, 1,99, 2,01 und die Zweierpotenzen 0,5, 1, 2: kein erreichtes Minimum mit S ≥ 2/η (für η ≥ 1 trivial); die Zweierpotenzen
    0,5 und 1 weichen nicht von den Nachbarwerten ab (767 gegen 780 und 764 konvergente Starts; 366 gegen 358 beschränkte), 2 gleicht der Seite 2,01."""
    rows = A.sharpness_over_etas(C.ETA_NEIGHBOURS)
    assert len(rows) == 9 and sum(r["too_sharp"] for r in rows) == 0
    by = {r["eta"]: r for r in rows}
    assert abs(by[0.5]["converged"] - by[0.49]["converged"]) < 20 and abs(by[1.0]["bounded"] - by[1.01]["bounded"]) < 20
    assert by[2.0]["bounded"] == by[2.01]["bounded"] == 0 and by[1.99]["bounded"] == 23


def test_random_starts_20000_never_reach_a_minimum_sharper_than_2_over_eta_and_never_converge_from_eta_1():
    """20 000 Zufallsstarts in [−5, 5]² (Zufalls-Seed 4242): η = 0,3: 8162 konvergieren, η = 0,7: 3367, η = 0,9: 2474, kein Minimum mit S ≥ 2/η; η = 1,01, 1,2, 1,5: keiner konvergiert."""
    r = A.random_start_counts((0.3, 0.7, 0.9, 1.01, 1.2, 1.5), 20_000, 4242)
    assert [r[e]["converged"] for e in (0.3, 0.7, 0.9)] == [8162, 3367, 2474]
    assert all(v["too_sharp"] == 0 for v in r.values())
    assert [r[e]["converged"] for e in (1.01, 1.2, 1.5)] == [0, 0, 0]
    assert [r[e]["bounded"] for e in (1.01, 1.2, 1.5)] == [2145, 1671, 1101]


@pytest.mark.parametrize("eta, per, n, mean, lo, hi", [(1.01, 2, 358, 2.005, 1.705, 2.305), (1.05, 2, 342, 2.025, 1.350, 2.700), (1.3, 16, 246, 2.369, 0.792, 4.107)])
def test_bounded_orbits_are_cycles_on_the_diagonal_and_eta_lambda_max_oscillates_around_2(eta, per, n, mean, lo, hi):
    """η = 1,01: 358 beschränkte Bahnen, alle auf der Diagonalen, alle Periode 2, Zeitmittel von η·λ_max 2,005, Bereich 1,705 bis 2,305; η = 1,05: 342, Periode 2, 2,025, 1,350 bis 2,700;
    η = 1,3: 246, Periode 16, 2,369, 0,792 bis 4,107; Anteil der Zeit über 2 im Median 0,5; Lyapunov-Exponenten negativ."""
    cl = K.classify_grid(eta)
    bp = K.bounded_periods(eta, cl)
    assert len(bp["period"]) == n and set(bp["period"].tolist()) == {per}
    assert np.all(np.abs(bp["x"] - bp["y"]) < 1e-6) and np.all(bp["lyapunov"] < 0)
    assert round(float(np.nanmedian(bp["mean_eta_lambda"])), 3) == mean
    assert (round(float(np.nanmin(bp["eta_lambda_min"])), 3), round(float(np.nanmax(bp["eta_lambda_max"])), 3)) == (lo, hi)
    assert float(np.nanmedian(bp["share_above_2"])) == pytest.approx(0.5, abs=1e-9)


def test_eta_1_5_bounded_orbits_are_all_chaotic_on_the_diagonal():
    """η = 1,5: 185 beschränkte Bahnen (6000 Schritte), alle auf der Diagonalen, keine Periode bis 128, alle mit positivem Lyapunov-Exponenten (Median +0,49), 715 divergieren."""
    r = A.efs_report(1.5)
    assert (r["n_bounded"], r["diverged"], r["on_diagonal"], r["periods"], r["n_chaotic"]) == (185, 715, 185, {0: 185}, 185)
    assert round(r["lyapunov_median"], 2) == 0.49


def test_doubling_thresholds_and_ratios():
    """Schwellen 1 (exakt), √5 − 1 = 1,23607 (exakt), 1,28803, 1,29923; Verhältnisse (η₂ − η₁)/(η₃ − η₂) = 4,54 und (η₃ − η₂)/(η₄ − η₃) = 4,64."""
    th = K.doubling_thresholds()
    assert [round(t, 5) for t in th] == [1.0, 1.23607, 1.28803, 1.29923]
    assert round((th[1] - th[0]) / (th[2] - th[1]), 2) == 4.54 and round((th[2] - th[1]) / (th[3] - th[2]), 2) == 4.64


def test_diagonal_grid_periods_and_chaos():
    """Gitter η = 0,9012 + 0,0125·k: Periode 1 bis 0,9887, Periode 2 von 1,0012 bis 1,2262, Periode 4 von 1,2387 bis 1,2762, Periode 8 bei 1,2887, Periode 16 bei 1,3012, Chaos von 1,3137 bis 1,4387,
    Periode 3 bei 1,4512, Periode 6 bei 1,4637 und 1,7012, Chaos von 1,4762 bis 1,6887 und von 1,7137 bis 1,7387."""
    got = [(round(a, 4), round(b, 4), kind) for a, b, kind in A.grid_ranges(A.diag_grid())]
    assert got == [(0.9012, 0.9887, "P1"), (1.0012, 1.2262, "P2"), (1.2387, 1.2762, "P4"), (1.2887, 1.2887, "P8"), (1.3012, 1.3012, "P16"), (1.3137, 1.4387, "chaos"),
                   (1.4512, 1.4512, "P3"), (1.4637, 1.4637, "P6"), (1.4762, 1.6887, "chaos"), (1.7012, 1.7012, "P6"), (1.7137, 1.7387, "chaos")]


def test_two_start_values_give_the_same_diagonal_type_on_the_grid():
    """Zwei Startwerte (t₀ = 1,2 und 0,3) geben auf dem Gitter überall dieselbe Art (68 Gitterpunkte)."""
    lo, step, n = C.DIAG_GRID
    for k in range(n):
        eta = lo + step * k
        a, b = K.diag_period(eta, 1.2), K.diag_period(eta, 0.3)
        assert (a[0] == b[0]) and (a[0] > 0 or (a[1] > 0.003) == (b[1] > 0.003)), eta


def test_newton_shares_in_the_disc_and_symmetry():
    """1 Mio Zufallspunkte in |z| ≤ 1,5 (Zufalls-Seed 11): Anteile 0,3339, 0,3338, 0,3324, jeder im Band 0,003 um 1/3; höchstens 1 von 10⁶ ohne Konvergenz; Symmetrie N(ωz) = ωN(z) auf 1e-13."""
    r = F.basin_shares(1_000_000, 1.5, 11)
    assert [round(s, 4) for s in r["shares"]] == [0.3339, 0.3338, 0.3324]
    assert all(abs(s - 1 / 3) < 0.003 for s in r["shares"]) and r["none"] <= 1e-6
    assert A.newton_report(2025)["symmetry_error"] < 1e-13


def test_newton_error_propagation_near_the_root():
    """Fehler e′ = e²: z₀ = 1,01 gibt 9,868·10⁻⁵ nach einem und 9,737·10⁻⁹ nach zwei Schritten (e′/e² = 0,9868 und 0,9999)."""
    e0 = 1e-2
    z = 1 + e0
    errs = []
    for _ in range(2):
        z = K.newton_step(z)
        errs.append(z - 1)
    assert errs[0] == pytest.approx(9.868e-5, rel=1e-3) and errs[1] == pytest.approx(9.737e-9, rel=1e-3)
    assert errs[0] / e0 ** 2 == pytest.approx(0.9868, abs=1e-4) and errs[1] / errs[0] ** 2 == pytest.approx(0.9999, abs=1e-4)


def test_every_preset_gives_the_expected_verdict_line():
    """Je Preset die erwartete Verdikt-Zeile: Handbeispiel konvergiert (Schärfe 2,7074 < 2/η), η = 1,01 ohne konvergenten Start (Periode 2), η = 1,3 Periode 16, η = 1,5 chaotisch."""
    kind, text = ev.verdict_step(A.step_report(3.0, 0.1, 0.3))
    assert kind == "success" and "9559" in text.replace(",", "") and "2,7074" in text
    kind, text = ev.verdict_sharp(K.classify_grid(1.01))
    assert kind == "success" and "konvergiert keiner" not in text and "kein stabiles Minimum" in text and "358 bleiben beschränkt" in text
    kind, text = ev.verdict_efs(A.efs_report(1.3))
    assert "Periode 16 (246 Bahnen)" in text and "2,369" in text and "Netze" in text
    th = K.doubling_thresholds()
    kind, text = ev.verdict_diag(A.diag_report(1.3), th, 1.3137)
    assert "Periode 16" in text and "kein Chaos" in text
    kind, text = ev.verdict_diag(A.diag_report(1.5), th, 1.3137)
    assert "chaotisch" in text and "1,3137" in text
    kind, text = ev.verdict_diag(A.diag_report(1.01), th, 1.3137)
    assert "Periode 2" in text and "chaotisch" not in text.replace("kein Chaos", "")


def test_the_handbook_start_converges_at_0_3_but_diverges_at_0_49_and_at_0_5():
    """Start (3; 0,1): η = 0,3 konvergiert (Schärfe 2,7074), η = 0,49 und η = 0,5 divergieren (der erste Schritt des Plans, 607/200 und 23/20, gehört zu η = 0,5; die Bahn läuft dann weg).
    Deshalb steht das Handbeispiel in der App bei η = 0,3."""
    status = {eta: A.step_report(3.0, 0.1, eta)["status"] for eta in (0.3, 0.49, 0.5)}
    assert status == {0.3: "converged", 0.49: "diverged", 0.5: "diverged"}
