"""Rechnet die Dimension der Grenze des Newton-Fraktals vor und friert sie als JSON ein (in App und README als „vorgerechnet“ gekennzeichnet).

    python tools/vorrechnen.py dimension     Unsicherheitsexponent (1,5 Mio Punkte) und Box-Zählung (Gitter 4001 x 4001), rund 1 bis 3 Minuten

Ergebnis: sdy_vorgerechnet_dimension.json neben den Modulen, mit Zufalls-Seed, Punktzahl und Laufzeit. tests/test_vorgerechnet.py rechnet kleine Fassungen nach."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sdy_constants as C  # noqa: E402
import sdy_fractal as F  # noqa: E402

POINTS = 1_500_000
PIXELS = 4001


def vorrechnen_dimension():
    t0 = time.time()
    shares = F.basin_shares(POINTS, C.NEWTON_RADIUS, C.DIMENSION_SEED)
    unc = F.uncertainty_dimension(POINTS, C.NEWTON_RADIUS, C.DIMENSION_SEED)
    lab, bd, h = F.boundary_map(PIXELS)
    wide = F.box_dimension(bd, h, use=slice(1, 6))
    fine = F.box_dimension(bd, h, use=slice(0, 4))
    out = {
        "beschreibung": "Newton-Fraktal z^3 = 1: Anteile der Einzugsgebiete in der Scheibe |z| <= 1,5, Unsicherheitsexponent und Box-Zählung der Grenze",
        "seed": C.DIMENSION_SEED, "punkte": POINTS, "radius": C.NEWTON_RADIUS, "anteile": shares["shares"], "ohne_konvergenz": shares["none"],
        "unsicherheit": {"eps": unc["eps"], "anteil": unc["fraction"], "alpha": unc["alpha"], "dimension": unc["dimension"]},
        "box": {"pixel": PIXELS, "halbe_breite": 1.5, "boxgroessen": wide["sizes"], "boxen_mit_rand": wide["counts"], "randpixel_anteil": wide["boundary_share"],
                "randpixel": int(bd.sum()), "dimension_4_bis_64": wide["dimension"], "dimension_2_bis_16": fine["dimension"],
                "anteile_im_quadrat": [float((lab == k).mean()) for k in range(3)]},
        "rechenzeit_sekunden": round(time.time() - t0),
    }
    (ROOT / "sdy_vorgerechnet_dimension.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Dimension: {out['rechenzeit_sekunden']} s -> sdy_vorgerechnet_dimension.json")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "dimension":
        sys.exit("Aufruf: python tools/vorrechnen.py dimension")
    vorrechnen_dimension()
