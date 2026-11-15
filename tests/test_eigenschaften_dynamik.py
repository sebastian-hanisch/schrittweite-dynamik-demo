"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, metamorph, 20 bis 50 Instanzen mit festem Zufalls-Seed): Aussagen, die für GD auf f = ½(xy − 1)² gelten müssen, ohne dass das Ergebnis bekannt ist.

Vertauschen von x und y vertauscht die Bahn; (x, y) → (−x, −y) spiegelt sie; die Diagonale bleibt invariant; Schärfe des erreichten Minimums = √(4 + D²); gleicher Zufalls-Seed gleiche
Ausgabe; die Reihenfolge der Starts ändert nichts; f ändert sich unter beiden Symmetrien nicht."""

import numpy as np
import pytest

import sdy_algorithm as A
import sdy_core as K
import sdy_fractal as F
from sdy_rng import SplitMix64


def _states(n, seed, box=3.0):
    rng = SplitMix64(seed)
    return [(2 * box * rng.random() - box, 2 * box * rng.random() - box, 0.05 + 1.5 * rng.random()) for _ in range(n)]


def test_swapping_x_and_y_swaps_the_orbit():
    for x, y, eta in _states(40, 1):
        px, py = K.gd_path(x, y, eta, 25)
        qx, qy = K.gd_path(y, x, eta, 25)
        assert np.allclose(px, qy, rtol=1e-12, atol=1e-12) and np.allclose(py, qx, rtol=1e-12, atol=1e-12)


def test_point_reflection_mirrors_the_orbit_and_keeps_the_loss():
    for x, y, eta in _states(40, 2):
        px, py = K.gd_path(x, y, eta, 25)
        qx, qy = K.gd_path(-x, -y, eta, 25)
        assert np.allclose(px, -qx, rtol=1e-12, atol=1e-12) and np.allclose(py, -qy, rtol=1e-12, atol=1e-12)
        assert K.loss(x, y) == pytest.approx(K.loss(-x, -y)) == pytest.approx(K.loss(y, x))


def test_the_diagonal_stays_invariant():
    for _, _, eta in _states(30, 3):
        t = 0.3 + 0.9 * (eta % 1)
        x = y = t
        for _ in range(30):
            x, y = K.gd_step(x, y, eta)
            assert x == y
            t = K.g_diag(t, eta)
            assert x == pytest.approx(t, rel=1e-9, abs=1e-9)


def test_sharpness_of_the_reached_minimum_is_sqrt_of_four_plus_d_squared():
    n = 0
    for x, y, eta in _states(50, 4):
        if eta >= 0.95:
            continue
        r = A.step_report(x, y, eta)
        if r["status"] == "converged" and not r["on_curve"]:
            n += 1
            assert r["sharp"] == pytest.approx(np.sqrt(4 + r["D_end"] ** 2), rel=1e-9) and r["sharp"] < 2 / eta
    assert n >= 10                                           # die Eigenschaft wurde an mindestens 10 konvergenten Bahnen geprüft


def test_d_shrinks_by_the_product_of_the_factors_over_a_whole_orbit():
    x, y, eta = 1.37, -0.52, 0.45
    d0, prod = x * x - y * y, 1.0
    for _ in range(60):
        prod *= K.diff_ratio(x, y, eta)
        x, y = K.gd_step(x, y, eta)
    assert x * x - y * y == pytest.approx(d0 * prod, abs=1e-10)


def test_order_of_the_starts_does_not_change_the_classification_and_same_seed_same_output():
    x0, y0 = K.grid_starts()
    perm = np.random.RandomState(0).permutation(len(x0))
    a = K.classify_grid(0.8)
    b = K.classify_grid(0.8, starts=(x0[perm], y0[perm]))
    assert (a["converged"], a["diverged"], a["bounded"]) == (b["converged"], b["diverged"], b["bounded"])
    assert np.array_equal(np.sort(a["sharp"]), np.sort(b["sharp"]))
    r1, r2 = A.random_start_counts((0.3,), 500, 9), A.random_start_counts((0.3,), 500, 9)
    assert r1 == r2 and F.basin_shares(2000, 1.5, 3) == F.basin_shares(2000, 1.5, 3) and F.basin_shares(2000, 1.5, 3) != F.basin_shares(2000, 1.5, 4)


def test_larger_step_size_gives_flatter_minima_on_average():
    """Über die 900 Starts sinkt die mittlere Schärfe der erreichten Minima mit η (η = 0,2, 0,51, 0,8); hier nur die Reihenfolge."""
    means = [float(np.mean(K.classify_grid(eta)["sharp"])) for eta in (0.2, 0.51, 0.8)]
    assert means[0] > means[1] > means[2]


def test_the_stability_threshold_is_two_over_sharpness_on_minima():
    """Ein Minimum der Schärfe S ist für GD stabil genau dann, wenn η < 2/S: Eigenwert 1 − η S im Betrag unter 1 (20 Minima, je eine Schrittweite darunter und darüber)."""
    rng = SplitMix64(6)
    for _ in range(20):
        x = 0.4 + 2.5 * rng.random()
        y = 1 / x
        s = K.sharpness(x, y)
        for eta, stable in ((0.97 * 2 / s, True), (1.03 * 2 / s, False)):
            assert (abs(1 - eta * s) < 1) == stable
            # eine Störung entlang des Gradienten wächst bzw. schrumpft entsprechend
            dx, dy = 1e-6 * y, 1e-6 * x
            x1, y1 = K.gd_step(x + dx, y + dy, eta)
            assert (np.hypot(x1 - x, y1 - y) < np.hypot(dx, dy)) == stable
