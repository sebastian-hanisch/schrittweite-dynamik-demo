"""Orakel (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12): jede Kernroutine gegen einen anderen Rechenweg.

Jacobi-Matrix per Differenzenquotient gegen die Hesse-Formel; Eigenwerte gegen `numpy.linalg.eigvalsh`; Zweierzyklen der Diagonalen gegen die Wurzeln von g(g(t)) − t per
`numpy.roots`; Verdopplungsschwellen (Brent) gegen die geschlossene Form √5 − 1 und gegen die Periode der Bahn auf beiden Seiten; Newton-Wurzeln gegen `numpy.roots`;
vektorisiertes Raster gegen eine skalare Schleife; Lyapunov-Exponent der Ebene gegen den der Diagonalabbildung."""

import math

import numpy as np
import pytest
from numpy.polynomial import Polynomial

import sdy_core as K
import sdy_fractal as F
from sdy_rng import SplitMix64


def test_hessian_eigenvalues_equal_numpy_and_the_finite_difference_jacobian():
    rng = SplitMix64(3)
    for _ in range(60):
        x, y, eta = 6 * rng.random() - 3, 6 * rng.random() - 3, 0.05 + 1.9 * rng.random()
        lo, hi = K.hessian_eigs(x, y)
        ref = np.linalg.eigvalsh(np.array([[y * y, 2 * x * y - 1], [2 * x * y - 1, x * x]]))
        assert (float(lo), float(hi)) == pytest.approx(tuple(ref), abs=1e-10)
        h, jac = 1e-6, np.zeros((2, 2))
        for j in range(2):
            d = np.zeros(2)
            d[j] = h
            jac[:, j] = (np.array(K.gd_step(x + d[0], y + d[1], eta)) - np.array(K.gd_step(x - d[0], y - d[1], eta))) / (2 * h)
        assert np.sort(np.linalg.eigvals(jac).real) == pytest.approx(np.sort(1 - eta * ref), abs=1e-5)


def test_identity_for_the_difference_of_squares_on_random_states():
    """x′² − y′² = (x² − y²)(1 − η²c²) an 500 Zufallszuständen (Abweichung unter 1e-12 relativ)."""
    rng = SplitMix64(5)
    for _ in range(500):
        x, y, eta = 6 * rng.random() - 3, 6 * rng.random() - 3, 0.05 + 1.95 * rng.random()
        x1, y1 = K.gd_step(x, y, eta)
        lhs, rhs = x1 ** 2 - y1 ** 2, (x ** 2 - y ** 2) * K.diff_ratio(x, y, eta)
        assert abs(lhs - rhs) / (1 + abs(rhs)) < 1e-12


@pytest.mark.parametrize("eta", [1.2, 1.1, 1.17, 1.01])
def test_two_cycles_equal_the_roots_of_g_of_g_minus_t(eta):
    g = Polynomial([0.0, 1 + eta, 0.0, -eta])
    roots = (g(g) - Polynomial([0.0, 1.0])).roots()
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
    a, b = K.two_cycle(eta)
    assert any(abs(a - r) < 1e-7 for r in real) and any(abs(b - r) < 1e-7 for r in real)
    sym = math.sqrt((2 + eta) / eta)
    assert any(abs(sym - r) < 1e-7 for r in real) and any(abs(-sym - r) < 1e-7 for r in real)


def test_doubling_thresholds_equal_the_closed_form_and_the_period_on_both_sides():
    th = K.doubling_thresholds()
    assert th[0] == 1.0 and th[1] == pytest.approx(math.sqrt(5) - 1, abs=1e-9)
    for (k, expected), eps in zip(((0, (1, 2)), (1, (2, 4)), (2, (4, 8)), (3, (8, 16))), (2e-3, 2e-3, 3e-4, 1e-4)):
        below, above = K.diag_period(th[k] - eps)[0], K.diag_period(th[k] + eps)[0]
        assert (below, above) == expected, (k, th[k])
    assert th == sorted(th) and th[2] == pytest.approx(1.2880317, abs=1e-6) and th[3] == pytest.approx(1.2992279, abs=1e-6)


def test_newton_roots_equal_numpy_roots_and_the_basin_is_stable_under_more_iterations():
    roots = np.roots([1, 0, 0, -1])
    assert sorted(np.round(roots, 12), key=lambda z: (z.real, z.imag)) == sorted(np.round(K.ROOTS, 12), key=lambda z: (z.real, z.imag))
    z = F.disc_points(2000, 1.5, 9)
    assert np.array_equal(K.newton_basin(z, 60), K.newton_basin(z, 200)) or np.mean(K.newton_basin(z, 60) != K.newton_basin(z, 200)) < 5e-3
    lab = K.newton_basin(z)
    near = np.abs(z - K.ROOTS[lab]) < 0.1
    assert np.all(lab[near & (lab >= 0)] == np.argmin(np.abs(z[near, None] - K.ROOTS), axis=1)[lab[near] >= 0]) or near.sum() == 0


def test_vectorised_grid_equals_a_scalar_loop_for_20_random_starts():
    rng = SplitMix64(8)
    x0 = np.array([6 * rng.random() - 3 for _ in range(20)])
    y0 = np.array([6 * rng.random() - 3 for _ in range(20)])
    for eta in (0.3, 0.8, 1.05, 1.3):
        cl = K.classify_grid(eta, starts=(x0, y0))
        for i in range(20):
            x, y, alive = float(x0[i]), float(y0[i]), True
            for _ in range(K.GRID_STEPS):
                x, y = K.gd_step(x, y, eta)
                if not (abs(x) < K.DIVERGE_LIMIT and abs(y) < K.DIVERGE_LIMIT):
                    alive = False
                    break
            status = "div" if not alive else ("conv" if K.loss(x, y) < K.LOSS_TOL else "bnd")
            got = "conv" if cl["conv"][i] else ("bnd" if cl["bnd"][i] else "div")
            assert got == status, (eta, i)


def test_lyapunov_of_a_converging_orbit_is_zero_and_on_the_chaotic_diagonal_it_matches_the_one_dimensional_exponent():
    """Konvergente Bahn: der größte Exponent ist 0 (neutrale Richtung entlang der Kurve der Minima; der zweite Multiplikator 1 − η·S ist kleiner); chaotische Diagonale bei η = 1,5:
    der Exponent der Ebene liegt bei dem der Diagonalabbildung (Tangente entlang der Diagonale, Querrichtung ist stabiler)."""
    eta = 0.3
    xs, ys = K.gd_path(3.0, 0.1, eta, 3000)
    lam = K.lyapunov_orbits(eta, xs[-1:], ys[-1:], 2000)[0]
    assert abs(lam) < 1e-3 and abs(1 - eta * K.sharpness(xs[-1], ys[-1])) < 1
    per, lam_diag = K.diag_period(1.5)
    assert per == 0 and lam_diag > 0.1
    t = 0.3
    for _ in range(3000):
        t = K.g_diag(t, 1.5)
    assert K.lyapunov_orbits(1.5, np.array([t]), np.array([t]), 4000)[0] == pytest.approx(lam_diag, abs=0.1)


def test_box_dimension_and_uncertainty_exponent_agree_within_a_band_on_small_versions():
    lab, bd, h = F.boundary_map(801)
    box = F.box_dimension(bd, h, sizes=(2, 4, 8, 16, 32), use=slice(0, 4))["dimension"]
    unc = F.uncertainty_dimension(60_000, 1.5, 3)["dimension"]
    assert 1.35 <= box <= 1.55 and 1.35 <= unc <= 1.55


def test_box_counting_on_a_known_set_gives_dimension_one_for_a_line_and_two_for_a_filled_square():
    """Orakel für die Box-Zählung selbst: eine Gerade hat die Dimension 1, eine gefüllte Fläche 2."""
    n = 1024
    line = np.zeros((n, n), bool)
    line[n // 2, :] = True
    full = np.ones((n, n), bool)
    sizes = (2, 4, 8, 16, 32)
    assert F.box_dimension(line, 1.0, sizes=sizes, use=slice(0, 5))["dimension"] == pytest.approx(1.0, abs=1e-9)
    assert F.box_dimension(full, 1.0, sizes=sizes, use=slice(0, 5))["dimension"] == pytest.approx(2.0, abs=1e-9)
