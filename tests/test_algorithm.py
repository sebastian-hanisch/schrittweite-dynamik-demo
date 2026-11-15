"""Einheiten des Kerns gegen Handrechnung (Handbeispiel des Plans, DEMO-PLAYBOOK §1b): jede Zahl steht als Rechnung mit exakten Brüchen, nicht als abgeschriebenes Ergebnis.

Das Handbeispiel rechnet mit η = 1/2 und η = 1: mit Brüchen unkritisch (kein Rundungsrauschen). Die Tests auf Gleitkomma nehmen Schrittweiten ohne Zweierpotenz (0,4, 0,45, 0,48, 1,2, 0,49)."""

import math
from fractions import Fraction

import numpy as np
import pytest

import sdy_algorithm as A
import sdy_core as K


def test_one_step_by_hand_with_fractions(hand):
    """Start (3; 1/10), η = 1/2: c = 3/10 − 1 = −7/10; x′ = 3 − ½·(1/10)·(−7/10) = 607/200; y′ = 1/10 − ½·3·(−7/10) = 23/20; Faktor 1 − η²c² = 1 − ¼·49/100 = 351/400."""
    x, y, eta = hand
    c = x * y - 1
    assert c == Fraction(-7, 10)
    x1, y1 = K.gd_step(x, y, eta)
    assert (x1, y1) == (Fraction(607, 200), Fraction(23, 20))
    assert K.diff_ratio(x, y, eta) == Fraction(351, 400)
    d0, d1 = x * x - y * y, x1 * x1 - y1 * y1
    assert d0 == Fraction(899, 100) and d1 == Fraction(7888725, 1000000) and d1 == d0 * Fraction(351, 400)


def test_sharpness_and_stability_of_the_minimum_2_one_half():
    """Minimum (2; ½): S = 4 + ¼ = 17/4, Hesse [[¼, 1], [1, 4]] hat die Eigenwerte 0 und 17/4; Eigenwerte des GD-Schritts 1 und 1 − η·17/4 (−0,7 bei η = 0,4: stabil; −1,125 bei η = 0,5: instabil)."""
    assert K.sharpness(2.0, 0.5) == 4.25
    lo, hi = K.hessian_eigs(2.0, 0.5)
    assert (float(lo), float(hi)) == pytest.approx((0.0, 4.25), abs=1e-12)
    for eta, expected in ((0.4, -0.7), (0.5, -1.125), (0.45, 1 - 0.45 * 4.25), (0.48, 1 - 0.48 * 4.25)):
        h = 1e-6                                           # Jacobi-Matrix per Differenzenquotient, unabhängig von der Hesse-Formel
        jac = np.zeros((2, 2))
        for j in range(2):
            d = np.zeros(2)
            d[j] = h
            plus = np.array(K.gd_step(2.0 + d[0], 0.5 + d[1], eta))
            minus = np.array(K.gd_step(2.0 - d[0], 0.5 - d[1], eta))
            jac[:, j] = (plus - minus) / (2 * h)
        ev = np.sort(np.linalg.eigvals(jac).real)
        assert ev[0] == pytest.approx(expected, abs=1e-6) and ev[1] == pytest.approx(1.0, abs=1e-6)
        assert (max(abs(ev)) <= 1 + 1e-9) == (eta < 8 / 17)    # stabil genau für η < 2/S = 8/17 = 0,470588


def test_diagonal_exact_fractions_eta_one():
    """η = 1, t₀ = 3/2: 3/2 → −3/8 → −357/512 → −141671523/134217728 (t → 2t − t³); g′(1) = 1 − 2η = −1."""
    t = Fraction(3, 2)
    seq = [t]
    for _ in range(3):
        t = K.g_diag(t, Fraction(1))
        seq.append(t)
    assert seq == [Fraction(3, 2), Fraction(-3, 8), Fraction(-357, 512), Fraction(-141671523, 134217728)]
    assert K.g_diag_prime(Fraction(1), Fraction(1)) == -1


def test_two_cycle_at_eta_1_2_and_at_the_second_doubling():
    """η = 1,2: a + b = √3,5, (a − b)² = 1/6, also a = 1,139538, b = 0,731290; g(a) = b; Multiplikator 9 − 2·2,2² = −0,68 (stabil). η = √5 − 1: Multiplikator −1, a = 1,144123, b = 1/√2."""
    a, b = K.two_cycle(1.2)
    assert (a, b) == pytest.approx((1.139538, 0.731290), abs=1e-6)
    assert K.g_diag(a, 1.2) == pytest.approx(b) and K.g_diag(b, 1.2) == pytest.approx(a)
    assert K.two_cycle_multiplier(1.2) == pytest.approx(-0.68) == pytest.approx(K.g_diag_prime(a, 1.2) * K.g_diag_prime(b, 1.2))
    eta = math.sqrt(5) - 1
    a, b = K.two_cycle(eta)
    assert (a, b) == pytest.approx((1.144123, 1 / math.sqrt(2)), abs=1e-6)
    assert K.two_cycle_multiplier(eta) == pytest.approx(-1.0, abs=1e-12)
    assert K.two_cycle_multiplier(1.0) == pytest.approx(1.0)


def test_symmetric_cycle_is_always_unstable():
    """η = 1: a = √3, g(√3) = 2√3 − 3√3 = −√3; Multiplikator (5 + 2η)² = 49."""
    a = math.sqrt(3)
    assert K.g_diag(a, 1.0) == pytest.approx(-a) and K.g_diag(-a, 1.0) == pytest.approx(a)
    assert K.g_diag_prime(a, 1.0) * K.g_diag_prime(-a, 1.0) == pytest.approx(49.0)


def test_newton_step_by_hand_and_symmetry():
    """N(11/10) = (2·1,331 + 1)/(3·1,21) = 1831/1815; Fehler 0,1 → 0,008815 (e² = 0,01, N″(1)/2 = 1). N(ωz) = ωN(z)."""
    assert K.newton_step(Fraction(11, 10)) == Fraction(1831, 1815)
    assert float(K.newton_step(Fraction(11, 10)) - 1) == pytest.approx(0.008815, abs=1e-6)
    z = np.array([0.7 + 0.2j, -1.3 + 0.9j, 0.05 - 1.1j])
    assert np.max(np.abs(K.newton_step(K.OMEGA * z) - K.OMEGA * K.newton_step(z))) < 1e-13


def test_step_report_of_the_handbook_example_with_eta_0_3():
    """Start (3; 0,1), η = 0,3 (Voreinstellung der App, keine Zweierpotenz): c = −0,7, x′ = 3 + 0,3·0,1·0,7 = 3,021, y′ = 0,1 + 0,3·3·0,7 = 0,73, Faktor 1 − 0,09·0,49 = 0,9559 (Brüche)."""
    eta, x, y = Fraction(3, 10), Fraction(3), Fraction(1, 10)
    c = x * y - 1
    x1, y1 = x - eta * y * c, y - eta * x * c
    assert (x1, y1, 1 - eta ** 2 * c ** 2) == (Fraction(3021, 1000), Fraction(73, 100), Fraction(9559, 10000))
    r = A.step_report(3.0, 0.1, 0.3)
    assert (r["c"], r["x1"], r["y1"], r["factor"]) == pytest.approx((-0.7, 3.021, 0.73, 0.9559))
    assert r["D1"] == pytest.approx(float(x1 * x1 - y1 * y1)) and r["D1"] == pytest.approx(r["D"] * r["factor"])
    assert r["status"] == "converged" and r["sharp"] < 2 / 0.3 and r["sharp"] == pytest.approx(math.sqrt(4 + r["D_end"] ** 2), rel=1e-9)
    assert r["sharp"] == pytest.approx(2.7074, abs=1e-3)       # Vormessung V5a: Schärfe 2,7074 gegen 2/η = 6,667
    assert r["steps"] is not None and r["steps"] > 5


def test_gd_path_and_grid_have_the_documented_shape():
    px, py = K.gd_path(3.0, 0.1, 0.3, 40)
    assert len(px) == len(py) == 41 and (px[0], py[0]) == (3.0, 0.1)
    x, y = K.grid_starts()
    assert len(x) == len(y) == 900 and not np.any(np.isclose(x, 0.5)) and x.min() == pytest.approx(-2.5 + 0.0137)
