"""Randfälle (DEMO-PLAYBOOK §12): η = 0, Start (0, 0), Start auf einer Achse, Start auf der Kurve x·y = 1, η größer 2, Start nahe der Hyperbel, Newton-Start z = 0 und auf den Symmetriestrahlen."""

import numpy as np
import pytest

import sdy_algorithm as A
import sdy_core as K


def test_eta_zero_keeps_the_orbit_still():
    x, y = K.gd_step(1.7, -0.4, 0.0)
    assert (x, y) == (1.7, -0.4)
    r = A.step_report(1.7, -0.4, 0.0)
    assert r["status"] == "bounded" and r["factor"] == 1.0 and r["period"] == 1


def test_start_at_the_origin_is_a_fixed_point_with_loss_one_half():
    """(0, 0): c = −1, der Gradient (y·c, x·c) ist null: Schritt null, die Bahn bleibt stehen, Verlust ½ (kein Minimum)."""
    assert K.gd_step(0.0, 0.0, 0.7) == (0.0, 0.0)
    r = A.step_report(0.0, 0.0, 0.7)
    assert r["status"] == "bounded" and r["period"] == 1 and K.loss(0.0, 0.0) == 0.5


def test_start_on_an_axis_stays_on_the_axis():
    """Auf der Achse y = 0 ist der Gradient (0, x·c) mit c = −1: y wächst, die Bahn verlässt die Achse; x bleibt fest."""
    x1, y1 = K.gd_step(2.3, 0.0, 0.3)
    assert x1 == 2.3 and y1 == pytest.approx(0.3 * 2.3)


@pytest.mark.parametrize("eta", [0.3, 0.9, 1.3])
def test_start_on_the_curve_of_minima_is_a_fixed_point_even_if_too_sharp(eta):
    """x·y = 1 ist ein Fixpunkt (c = 0) auch mit S ≥ 2/η; solche Starts (|x y − 1| < 1e-9) werden aus der Zählung ausgenommen."""
    x, y = 4.0, 0.25                                         # S = 16,0625 ≫ 2/η
    assert K.gd_step(x, y, eta) == (x, y)
    cl = K.classify_grid(eta, starts=(np.array([x]), np.array([y])))
    assert cl["converged"] == 1 and cl["too_sharp"] == 0 and cl["sharp_max"] > cl["limit"]
    assert "Kurve x·y = 1" in __import__("sdy_evaluation").verdict_step(A.step_report(x, y, eta))[1]


def test_eta_above_two_diverges_for_every_start():
    for eta in (2.05, 2.1, 3.0):
        cl = K.classify_grid(eta)
        assert cl["diverged"] == 900 and cl["converged"] == 0 and cl["bounded"] == 0


def test_start_next_to_the_hyperbola_converges_for_small_eta():
    r = A.step_report(2.0, 0.5 + 1e-7, 0.3)
    assert r["status"] == "converged" and r["sharp"] == pytest.approx(4.25, abs=1e-3)


def test_newton_start_at_zero_is_excluded_and_symmetry_rays_converge():
    """z = 0: Division durch null, nicht konvergiert (−1). Die drei Symmetriestrahlen (positive reelle Achse und die um 120° und 240° gedrehten) laufen zur Wurzel auf dem Strahl."""
    assert K.newton_basin(np.array([0j]))[0] == -1
    r = np.array([0.4, 1.7, 3.0])
    assert list(K.newton_basin(r)) == [0, 0, 0]
    assert list(K.newton_basin(r * K.OMEGA)) == [1, 1, 1] and list(K.newton_basin(r * K.OMEGA.conjugate())) == [2, 2, 2]


def test_newton_basin_of_empty_and_single_point_inputs():
    assert K.newton_basin(np.array([], dtype=complex)).shape == (0,)
    assert K.newton_basin(np.array([[1.2 + 0.1j]])).shape == (1, 1)


def test_bounded_periods_of_an_empty_set():
    cl = K.classify_grid(0.3)
    assert cl["bounded"] == 0
    bp = K.bounded_periods(0.3, cl)
    assert len(bp["period"]) == 0 and len(bp["lyapunov"]) == 0
    assert A.efs_report(0.3)["n_bounded"] == 0


def test_two_cycle_does_not_exist_up_to_eta_one():
    with pytest.raises(ValueError):
        K.two_cycle(1.0)
