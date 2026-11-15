"""Kern: Gradientenabstieg (GD) auf f(x, y) = 1/2 (x y - 1)^2, die Diagonalabbildung t -> (1 + eta) t - eta t^3, Newton-Verfahren für z^3 = 1.

Modell von Hand gewählt (keine Quelle): Minima auf der Kurve x y = 1 mit Schärfe S = x^2 + y^2 >= 2 (größter Eigenwert der Hesse-Matrix dort).
Alle Schrittfunktionen rechnen mit gewöhnlicher Arithmetik und akzeptieren deshalb auch `fractions.Fraction` (Handbeispiele und Tests mit exakten Brüchen)."""

import math

import numpy as np

np.seterr(all="ignore")

DIVERGE_LIMIT = 1e8
LOSS_TOL = 1e-12          # „konvergiert“: Verlust unter 1e-12 nach den Schritten
GRID_N = 30
GRID_LO, GRID_HI = -2.5, 2.5
GRID_OFFSET = (0.0137, -0.0291)   # Versatz: kein Startwert liegt auf einem Rasterpunkt wie 0,5
GRID_STEPS = 6000
PERIOD_MAX = 128
LYAPUNOV_CHAOS = 0.003    # „Chaos“ erst über diesem Lyapunov-Exponenten (nie bei Perioden)


# ----------------------------------------------------------------------------------------------------------------------------------- Ebene
def loss(x, y):
    return 0.5 * (x * y - 1) ** 2


def gd_step(x, y, eta):
    """Ein GD-Schritt: Gradient (y c, x c) mit c = x y - 1."""
    c = x * y - 1
    return x - eta * y * c, y - eta * x * c


def sharpness(x, y):
    """Schärfe S = x^2 + y^2: größter Eigenwert der Hesse-Matrix auf der Kurve x y = 1 (der andere ist 0)."""
    return x * x + y * y


def hessian_eigs(x, y):
    """Eigenwerte der Hesse-Matrix [[y^2, 2xy - 1], [2xy - 1, x^2]] (kleiner, größer), geschlossene Formel der 2x2-Matrix."""
    h11, h22, h12 = y * y, x * x, 2 * x * y - 1
    root = np.sqrt((h11 - h22) ** 2 + 4 * h12 ** 2)
    return 0.5 * (h11 + h22 - root), 0.5 * (h11 + h22 + root)


def diff_ratio(x, y, eta):
    """Faktor 1 - eta^2 c^2, mit dem D = x^2 - y^2 in einem Schritt schrumpft (Identität x'^2 - y'^2 = (x^2 - y^2)(1 - eta^2 c^2))."""
    c = x * y - 1
    return 1 - eta ** 2 * c ** 2


def gd_path(x, y, eta, n):
    """Die ersten n Schritte der Bahn (n + 1 Punkte); bei Überlauf bricht die Bahn mit dem letzten endlichen Punkt ab."""
    xs, ys = [x], [y]
    for _ in range(n):
        x, y = gd_step(x, y, eta)
        if not (abs(x) < DIVERGE_LIMIT and abs(y) < DIVERGE_LIMIT):
            break
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def grid_starts(n=GRID_N, lo=GRID_LO, hi=GRID_HI, offset=GRID_OFFSET):
    g = np.linspace(lo, hi, n)
    gx, gy = np.meshgrid(g, g)
    return gx.ravel() + offset[0], gy.ravel() + offset[1]


def run_grid(eta, steps=GRID_STEPS, starts=None):
    """GD von vielen Startwerten zugleich. Rückgabe: Endzustände x, y, Bahn lebt (nicht divergiert), Zustand."""
    x, y = starts if starts is not None else grid_starts()
    x, y = x.copy(), y.copy()
    alive = np.ones(len(x), bool)
    for _ in range(steps):
        g = x * y - 1.0
        xn, yn = x - eta * g * y, y - eta * g * x
        x, y = np.where(alive, xn, x), np.where(alive, yn, y)
        alive &= (np.abs(x) < DIVERGE_LIMIT) & (np.abs(y) < DIVERGE_LIMIT)
    return x, y, alive


def classify_grid(eta, steps=GRID_STEPS, starts=None):
    """Zählung: konvergiert / divergiert / beschränkt ohne Konvergenz, Schärfe der erreichten Minima, Zahl der Minima mit S >= 2/eta (ohne Starts auf der Kurve x y = 1)."""
    x0, y0 = starts if starts is not None else grid_starts()
    x, y, alive = run_grid(eta, steps, (x0, y0))
    ls = loss(x, y)
    conv = alive & (ls < LOSS_TOL)
    bnd = alive & ~conv
    sharp = sharpness(x, y)[conv]
    on_curve = np.abs(x0 * y0 - 1) < 1e-9
    limit = 2.0 / eta if eta > 0 else float("inf")
    return {
        "eta": eta, "n": len(x0), "converged": int(conv.sum()), "diverged": int((~alive).sum()), "bounded": int(bnd.sum()),
        "sharp": sharp, "sharp_max": float(sharp.max()) if len(sharp) else float("nan"), "limit": limit,
        "too_sharp": int(np.sum(sharpness(x, y)[conv & ~on_curve] >= limit)), "x": x, "y": y, "conv": conv, "bnd": bnd,
    }


def bounded_periods(eta, cl, pmax=PERIOD_MAX):
    """Perioden der beschränkten Bahnen (kleinstes p <= pmax mit Zustand nach p Schritten gleich dem jetzigen; 0 = keine) und Zeitmittel von eta * lambda_max(Hesse) über eine Periode."""
    x, y = cl["x"][cl["bnd"]], cl["y"][cl["bnd"]]
    n = len(x)
    per = np.zeros(n, int)
    hist = [(x.copy(), y.copy())]
    xx, yy = x.copy(), y.copy()
    for _ in range(pmax):
        xx, yy = gd_step(xx, yy, eta)
        hist.append((xx.copy(), yy.copy()))
    for p in range(1, pmax + 1):
        same = (np.abs(hist[p][0] - x) < 1e-9) & (np.abs(hist[p][1] - y) < 1e-9) & (per == 0)
        per[same] = p
    # eta * lambda_max entlang einer Periode: exakter Zeitmittelwert (die Bahn ist periodisch)
    mean_e = np.full(n, np.nan)
    above = np.full(n, np.nan)
    emin = np.full(n, np.nan)
    emax = np.full(n, np.nan)
    for p in sorted(set(int(v) for v in per if v > 0)):
        sel = np.nonzero(per == p)[0]
        e = _efs_series(x[sel], y[sel], eta, p)
        mean_e[sel] = e.mean(axis=0)
        above[sel] = (e > 2.0).mean(axis=0)
        emin[sel], emax[sel] = e.min(axis=0), e.max(axis=0)
    aper = np.nonzero(per == 0)[0]
    if len(aper):                                          # ohne Periode bis pmax: Mittel über 4000 Schritte (Näherung, kein exakter Wert)
        e = _efs_series(x[aper], y[aper], eta, 4000)
        mean_e[aper], above[aper] = e.mean(axis=0), (e > 2.0).mean(axis=0)
        emin[aper], emax[aper] = e.min(axis=0), e.max(axis=0)
    lam = lyapunov_orbits(eta, x, y, 4000) if n else np.empty(0)
    return {"period": per, "mean_eta_lambda": mean_e, "share_above_2": above, "eta_lambda_min": emin, "eta_lambda_max": emax, "lyapunov": lam, "x": x, "y": y}


def _efs_series(x, y, eta, n):
    """eta * lambda_max(Hesse) entlang von n Schritten für viele Bahnen: Feld (n, Zahl der Bahnen)."""
    xs, ys = x.copy(), y.copy()
    out = np.empty((n, len(xs)))
    for k in range(n):
        out[k] = eta * hessian_eigs(xs, ys)[1]
        xs, ys = gd_step(xs, ys, eta)
    return out


def lyapunov_orbits(eta, x, y, n=4000):
    """Größter Lyapunov-Exponent je Bahn (Benettin: Tangentenvektor mit der Jacobi-Matrix 1 - eta * Hesse mitführen, jeden Schritt normieren)."""
    xx, yy = np.array(x, float), np.array(y, float)
    v0, v1 = np.ones(len(xx)), np.zeros(len(xx))
    acc = np.zeros(len(xx))
    for _ in range(n):
        c = xx * yy - 1.0
        h11, h22, h12 = yy * yy, xx * xx, 2 * xx * yy - 1.0
        a0 = v0 - eta * (h11 * v0 + h12 * v1)
        a1 = v1 - eta * (h12 * v0 + h22 * v1)
        nv = np.sqrt(a0 ** 2 + a1 ** 2)
        acc += np.log(nv)
        v0, v1 = a0 / nv, a1 / nv
        xx, yy = xx - eta * c * yy, yy - eta * c * xx
    return acc / n


# ----------------------------------------------------------------------------------------------------------------------------------- Diagonale
def g_diag(t, eta):
    """GD auf der Diagonale x = y = t: t -> (1 + eta) t - eta t^3."""
    return (1 + eta) * t - eta * t ** 3


def g_diag_prime(t, eta):
    return 1 + eta - 3 * eta * t * t


def diag_orbit(eta, t0=1.2, n=60):
    t = [t0]
    for _ in range(n):
        if abs(t[-1]) > 1e6:          # divergiert: die Bahn endet hier (kein Überlauf)
            break
        t.append(g_diag(t[-1], eta))
    return np.array(t)


def two_cycle(eta):
    """Der Zweierzyklus (a, b) mit a + b = sqrt(1 + 3/eta), a b = 1/eta, (a - b)^2 = 1 - 1/eta (existiert für eta > 1)."""
    if eta <= 1:
        raise ValueError("der Zweierzyklus (a, b) existiert nur für eta > 1")
    s, d = math.sqrt(1 + 3 / eta), math.sqrt(1 - 1 / eta)
    return (s + d) / 2, (s - d) / 2


def two_cycle_multiplier(eta):
    """Multiplikator g'(a) g'(b) des Zweierzyklus = 9 - 2 (1 + eta)^2 (geschlossen; gleich 1 bei eta = 1, gleich -1 bei eta = sqrt5 - 1)."""
    return 9 - 2 * (1 + eta) ** 2


def bifurcation(etas, burn=3000, keep=80, t0=1.2):
    """Bifurkationsdiagramm und Lyapunov-Exponent der Diagonalabbildung für viele eta zugleich. Rückgabe: Punkte (eta, t) und lambda je eta (über `burn` Schritte nach dem Einschwingen)."""
    etas = np.asarray(etas, dtype=float)
    t = np.full(len(etas), t0)
    for _ in range(burn):
        t = g_diag(t, etas)
    pts_eta, pts_t = [], []
    lam = np.zeros(len(etas))
    for i in range(burn):
        lam += np.log(np.abs(g_diag_prime(t, etas)) + 1e-300)
        t = g_diag(t, etas)
        if i >= burn - keep:
            pts_eta.append(etas.copy())
            pts_t.append(t.copy())
    return np.concatenate(pts_eta), np.concatenate(pts_t), lam / burn


def diag_period(eta, t0=1.2, burn=60000, pmax=PERIOD_MAX, tol=1e-9):
    """Periode der Bahn der Diagonalabbildung nach dem Einschwingen (0 = keine bis pmax) und Lyapunov-Exponent über 20 000 Schritte."""
    t = t0
    for _ in range(burn):
        t = g_diag(t, eta)
        if abs(t) > 1e6:
            return None, float("nan")
    ref, tt, per = t, t, 0
    for p in range(1, pmax + 1):
        tt = g_diag(tt, eta)
        if abs(tt - ref) < tol:
            per = p
            break
    lam, tt = 0.0, t
    for _ in range(20000):
        lam += math.log(abs(g_diag_prime(tt, eta)) + 1e-300)
        tt = g_diag(tt, eta)
    return per, lam / 20000


def _cycle_multiplier(eta, p, t0):
    t = t0
    for _ in range(60):
        gt, d = t, 1.0
        for _ in range(p):
            d *= g_diag_prime(gt, eta)
            gt = g_diag(gt, eta)
        step = (gt - t) / (d - 1.0)
        t -= step
        if abs(step) < 1e-14:
            break
    d, gt = 1.0, t
    for _ in range(p):
        d *= g_diag_prime(gt, eta)
        gt = g_diag(gt, eta)
    return d


def doubling_thresholds():
    """Schwellen eta_1 = 1 (exakt), eta_2 = sqrt5 - 1 (exakt, hier per Brent aus dem Multiplikator = -1 gefunden), eta_3, eta_4 (nur numerisch)."""
    from scipy.optimize import brentq

    def attractor_point(eta):
        t = 1.2
        for _ in range(60000):
            t = g_diag(t, eta)
        return t

    out = [1.0]
    for p, (lo, hi) in ((2, (1.20, 1.2387)), (4, (1.2762, 1.2887)), (8, (1.2887, 1.3012))):
        t0 = attractor_point(lo)
        out.append(brentq(lambda e, p=p, t0=t0: _cycle_multiplier(e, p, t0) + 1.0, lo, hi, xtol=1e-13))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------- Newton
OMEGA = complex(-0.5, math.sqrt(3) / 2)
ROOTS = np.array([1.0 + 0j, OMEGA, OMEGA.conjugate()])


def newton_step(z):
    """N(z) = z - (z^3 - 1)/(3 z^2) = (2 z^3 + 1)/(3 z^2)."""
    return (2 * z ** 3 + 1) / (3 * z ** 2)


def newton_basin(z, iters=60, chunk=1 << 20):
    """Index der Wurzel (0, 1, 2), gegen die Newton von z konvergiert; -1, wenn nach `iters` Schritten keine Wurzel auf 1e-6 erreicht ist (z = 0 gehört dazu).
    Große Felder werden in Stücken gerechnet (Speicher)."""
    z = np.asarray(z, dtype=complex)
    shape = z.shape
    flat = z.ravel()
    out = np.empty(flat.shape, dtype=np.int8)
    for a in range(0, len(flat), chunk):
        w = flat[a:a + chunk].copy()
        for _ in range(iters):
            w = w - (w ** 3 - 1) / (3 * w ** 2)
        d = np.abs(w[:, None] - ROOTS)
        out[a:a + chunk] = np.where(np.min(d, axis=1) < 1e-6, np.argmin(d, axis=1), -1)
    return out.reshape(shape).astype(int)
