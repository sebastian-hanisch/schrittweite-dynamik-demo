"""Die vorgerechnete Dimensionsmessung (tools/vorrechnen.py dimension) ist keine eingefrorene Behauptung: Struktur, Seeds und kleine Fassungen werden neu gerechnet; der Vollauf (66 s) steht nicht im Test."""

import numpy as np
import pytest

import sdy_constants as C
import sdy_evaluation as ev
import sdy_fractal as F

data = ev.load_dimension()
pytestmark = pytest.mark.skipif(data is None, reason="sdy_vorgerechnet_dimension.json fehlt")


def test_structure_and_parameters():
    assert data["seed"] == C.DIMENSION_SEED and data["punkte"] == 1_500_000 and data["radius"] == C.NEWTON_RADIUS
    assert data["box"]["pixel"] == 4001 and data["box"]["boxgroessen"] == [2, 4, 8, 16, 32, 64, 128]
    assert len(data["unsicherheit"]["eps"]) == len(data["unsicherheit"]["anteil"]) == 6


def test_shares_and_dimensions_are_consistent_with_their_own_data():
    assert all(abs(s - 1 / 3) < 0.003 for s in data["anteile"]) and abs(sum(data["anteile"]) + data["ohne_konvergenz"] - 1) < 1e-9
    eps, frac = np.array(data["unsicherheit"]["eps"]), np.array(data["unsicherheit"]["anteil"])
    slope = np.polyfit(np.log(eps), np.log(frac), 1)[0]
    assert slope == pytest.approx(data["unsicherheit"]["alpha"]) and 2 - slope == pytest.approx(data["unsicherheit"]["dimension"])
    s, c = data["box"]["boxgroessen"], data["box"]["boxen_mit_rand"]
    h = 2 * data["box"]["halbe_breite"] / (data["box"]["pixel"] - 1)
    assert -np.polyfit(np.log([x * h for x in s[1:6]]), np.log(c[1:6]), 1)[0] == pytest.approx(data["box"]["dimension_4_bis_64"])
    assert -np.polyfit(np.log([x * h for x in s[0:4]]), np.log(c[0:4]), 1)[0] == pytest.approx(data["box"]["dimension_2_bis_16"])


def test_the_numbers_of_the_readme():
    """Unsicherheitsexponent 0,553 (D = 1,447), Box-Zählung 1,454 und 1,477, 272 386 Randpixel."""
    assert round(data["unsicherheit"]["alpha"], 3) == 0.553 and round(data["unsicherheit"]["dimension"], 3) == 1.447
    assert (round(data["box"]["dimension_4_bis_64"], 3), round(data["box"]["dimension_2_bis_16"], 3), data["box"]["randpixel"]) == (1.454, 1.477, 272386)


def test_both_methods_agree_within_the_band_1_40_to_1_52():
    d = (data["unsicherheit"]["dimension"], data["box"]["dimension_4_bis_64"], data["box"]["dimension_2_bis_16"])
    assert all(1.40 <= v <= 1.52 for v in d)


def test_the_boundary_share_and_the_square_shares_of_the_grid():
    box = data["box"]
    assert box["randpixel"] / box["pixel"] ** 2 == pytest.approx(box["randpixel_anteil"])
    assert sum(box["anteile_im_quadrat"]) == pytest.approx(1.0, abs=1e-9)
    assert [round(100 * s, 1) for s in box["anteile_im_quadrat"]] == [34.5, 32.7, 32.7]      # im Quadrat (nicht in der Scheibe) ist das Gebiet der 1 größer


def test_small_recomputation_reproduces_the_dimension_within_a_band():
    """Kleine Fassung: 801 × 801 Gitter und 60 000 Punkte geben dieselbe Dimension innerhalb von 0,08."""
    lab, bd, h = F.boundary_map(801)
    small = F.box_dimension(bd, h, sizes=(2, 4, 8, 16, 32), use=slice(0, 4))["dimension"]
    assert abs(small - data["box"]["dimension_2_bis_16"]) < 0.08
    assert abs(F.uncertainty_dimension(60_000, 1.5, 11)["dimension"] - data["unsicherheit"]["dimension"]) < 0.08
