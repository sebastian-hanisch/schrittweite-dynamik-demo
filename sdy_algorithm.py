"""Messroutinen der Experimente (rein: gleiche Eingabe, gleiche Ausgabe): Schritt und Landschaft, Schärfe gegen 2/eta, Diagonale, eta·lambda_max, Newton.

Die Rechenkerne stehen in sdy_core.py (GD, Diagonalabbildung, Newton) und sdy_fractal.py (Anteile, Dimension)."""

import math

import numpy as np

import sdy_constants as C
import sdy_core as K
import sdy_fractal as F


# ----------------------------------------------------------------------------------------------------------------------------- Experiment 1
def steps_to_converge(x: float, y: float, eta: float, limit: int = C.GRID_STEPS):
    """Zahl der Schritte, bis der Verlust unter 1e-12 liegt; None, wenn das innerhalb von `limit` Schritten nicht eintritt (Divergenz oder beschränkte Bahn)."""
    for k in range(limit + 1):
        if K.loss(x, y) < K.LOSS_TOL:
            return k
        x, y = K.gd_step(x, y, eta)
        if not (abs(x) < K.DIVERGE_LIMIT and abs(y) < K.DIVERGE_LIMIT):
            return None
    return None


def step_report(x: float, y: float, eta: float) -> dict:
    """Ein Schritt von (x, y) und das Ende der Bahn nach `GRID_STEPS` Schritten: Ergebnis (konvergiert, divergiert, beschränkt), Minimum, Schärfe, Periode."""
    c = x * y - 1
    x1, y1 = K.gd_step(x, y, eta)
    d0, d1 = x * x - y * y, x1 * x1 - y1 * y1
    cl = K.classify_grid(eta, starts=(np.array([x]), np.array([y])))
    status = "converged" if cl["converged"] else "diverged" if cl["diverged"] else "bounded"
    xe, ye = float(cl["x"][0]), float(cl["y"][0])
    rep = {"x": x, "y": y, "eta": eta, "c": c, "x1": x1, "y1": y1, "D": d0, "D1": d1, "factor": K.diff_ratio(x, y, eta), "status": status, "on_curve": abs(c) < 1e-9,
           "xe": xe, "ye": ye, "limit": 2.0 / eta if eta > 0 else float("inf"), "period": None, "steps": None}
    if status == "converged":
        rep["sharp"] = K.sharpness(xe, ye)
        rep["D_end"] = xe * xe - ye * ye
        rep["steps"] = steps_to_converge(x, y, eta)
    elif status == "bounded":
        rep["period"] = int(K.bounded_periods(eta, cl)["period"][0])
    return rep


# ----------------------------------------------------------------------------------------------------------------------------- Experiment 2
def sharpness_counts(eta: float) -> dict:
    """900 Starts: Zählung und Schärfen der erreichten Minima (siehe sdy_core.classify_grid)."""
    return K.classify_grid(eta)


def sharpness_over_etas(etas) -> list:
    """Je Schrittweite: konvergiert, divergiert, beschränkt, größte Schärfe, Zahl der Minima mit S >= 2/eta."""
    out = []
    for eta in etas:
        cl = K.classify_grid(eta)
        out.append({k: cl[k] for k in ("eta", "converged", "diverged", "bounded", "sharp_max", "limit", "too_sharp")})
    return out


def random_start_counts(etas, n: int, seed: int, box: float = 5.0) -> dict:
    """Gegenprobe der Absolutaussagen mit n Zufallsstarts in [-box, box]^2: je eta Zahl der Minima mit S >= 2/eta (Starts mit |x y - 1| < 1e-9 ausgenommen) und der Konvergenzen."""
    from sdy_rng import SplitMix64
    rng = SplitMix64(seed)
    x0 = (2 * rng.uniform_array(n) - 1) * box
    y0 = (2 * rng.uniform_array(n) - 1) * box
    out = {}
    for eta in etas:
        cl = K.classify_grid(eta, starts=(x0, y0))
        out[eta] = {"converged": cl["converged"], "too_sharp": cl["too_sharp"], "diverged": cl["diverged"], "bounded": cl["bounded"], "n": n}
    return out


# ----------------------------------------------------------------------------------------------------------------------------- Experiment 3
def diag_report(eta: float) -> dict:
    """Periode (0 = keine bis 128, None = divergiert) und Lyapunov-Exponent der Diagonalabbildung bei eta, dazu die ersten 60 Werte der Bahn ab t0 = 1,2."""
    per, lam = K.diag_period(eta, C.DIAG_T0)
    return {"eta": eta, "period": per, "lyapunov": lam, "orbit": K.diag_orbit(eta, C.DIAG_T0, 60)}


def diag_kind(rep: dict) -> str:
    """„P<k>“ für Perioden, „chaos“ nur bei Lyapunov-Exponent über 0,003 und ohne Periode, „div“ bei Divergenz, sonst „?“."""
    if rep["period"] is None:
        return "div"
    if rep["period"] > 0:
        return f"P{rep['period']}"
    return "chaos" if rep["lyapunov"] > K.LYAPUNOV_CHAOS else "?"


def diag_grid() -> list:
    """Gitter der Vormessung eta = 0,9012 + 0,0125 k (k = 0 .. 67): Art der Bahn je Schrittweite."""
    lo, step, n = C.DIAG_GRID
    rows = []
    for k in range(n):
        eta = lo + step * k
        rows.append({"eta": eta, "kind": diag_kind(diag_report(eta))})
    return rows


def grid_ranges(rows: list) -> list:
    """Zusammenhängende Bereiche gleicher Art: [(eta_von, eta_bis, Art), ...]."""
    out = []
    for r in rows:
        if out and out[-1][2] == r["kind"]:
            out[-1] = (out[-1][0], r["eta"], r["kind"])
        else:
            out.append((r["eta"], r["eta"], r["kind"]))
    return out


# ----------------------------------------------------------------------------------------------------------------------------- Experiment 4
def efs_report(eta: float) -> dict:
    """Beschränkte Bahnen ohne Konvergenz im 900er Raster und eta·lambda_max(Hesse) entlang der Bahn (exaktes Zeitmittel über eine Periode, bei fehlender Periode Mittel über 4000 Schritte)."""
    cl = K.classify_grid(eta)
    if cl["bounded"] == 0:
        return {"eta": eta, "n_bounded": 0, "converged": cl["converged"], "diverged": cl["diverged"]}
    bp = K.bounded_periods(eta, cl)
    per = bp["period"]
    vals, cnt = np.unique(per, return_counts=True)
    first = int(np.argmax(per > 0)) if (per > 0).any() else 0
    series_len = int(per[first]) * max(1, 40 // max(1, int(per[first]))) if per[first] > 0 else 40
    series = K._efs_series(bp["x"][first:first + 1], bp["y"][first:first + 1], eta, series_len)[:, 0]
    return {
        "eta": eta, "n_bounded": int(cl["bounded"]), "converged": cl["converged"], "diverged": cl["diverged"], "periods": dict(zip(vals.tolist(), cnt.tolist())),
        "on_diagonal": int(np.sum(np.abs(bp["x"] - bp["y"]) < 1e-6)), "lyapunov_median": float(np.median(bp["lyapunov"])), "n_chaotic": int(np.sum(bp["lyapunov"] > K.LYAPUNOV_CHAOS)),
        "mean_median": float(np.nanmedian(bp["mean_eta_lambda"])), "range_min": float(np.nanmin(bp["eta_lambda_min"])), "range_max": float(np.nanmax(bp["eta_lambda_max"])),
        "above_median": float(np.nanmedian(bp["share_above_2"])), "series": series, "exact": bool((per > 0).all()),
    }


# ----------------------------------------------------------------------------------------------------------------------------- Experiment 5
def newton_report(seed: int, n: int = C.NEWTON_POINTS) -> dict:
    """Anteile der Einzugsgebiete in der Scheibe |z| <= 1,5 (n Zufallspunkte, Zufalls-Seed) und die Symmetrie N(w z) = w N(z) an 1000 Punkten."""
    shares = F.basin_shares(n, C.NEWTON_RADIUS, seed)
    z = F.disc_points(1000, 2.0, seed + 5)
    sym = float(np.max(np.abs(K.newton_step(K.OMEGA * z) - K.OMEGA * K.newton_step(z))))
    return {**shares, "symmetry_error": sym}


def newton_image(center: complex, half: float, n: int = C.IMAGE_N):
    return F.image(center, half, n)


def dimension_summary(data: dict) -> dict:
    """Beide Dimensionsmessungen aus der vorgerechneten Datei: Unsicherheitsexponent und Box-Zählung (zwei Boxbereiche)."""
    return {"uncertainty": data["unsicherheit"]["dimension"], "box_wide": data["box"]["dimension_4_bis_64"], "box_fine": data["box"]["dimension_2_bis_16"]}
