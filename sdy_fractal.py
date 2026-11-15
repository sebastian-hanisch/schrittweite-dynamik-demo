"""Newton-Fraktal z^3 = 1: Anteile der Einzugsgebiete, Dimension der Grenze (Unsicherheitsexponent und Box-Zählung), Bild.

Zufall nur über SplitMix64 (sdy_rng.py). Die Dimension ist eine Messung mit zwei Verfahren, keine Literaturgegenprobe."""

import math

import numpy as np

import sdy_core as K
from sdy_rng import SplitMix64


def disc_points(n: int, radius: float, seed: int) -> np.ndarray:
    """n Zufallspunkte gleichverteilt in der Kreisscheibe |z| <= radius."""
    rng = SplitMix64(seed)
    rad = radius * np.sqrt(rng.uniform_array(n))
    ang = 2 * math.pi * rng.uniform_array(n)
    return rad * np.exp(1j * ang)


def basin_shares(n: int, radius: float, seed: int) -> dict:
    """Anteile der drei Einzugsgebiete in der Scheibe |z| <= radius (n Zufallspunkte) und Anteil ohne Konvergenz."""
    lab = K.newton_basin(disc_points(n, radius, seed))
    return {"n": n, "radius": radius, "shares": [float(np.mean(lab == k)) for k in range(3)], "none": float(np.mean(lab == -1))}


EPS_LIST = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)


def uncertainty_dimension(n: int, radius: float, seed: int, eps_list=EPS_LIST) -> dict:
    """Unsicherheitsexponent alpha: Anteil der Punkte, deren Gebiet sich bei einer Störung vom Betrag eps ändert, wächst wie eps^alpha; Dimension der Grenze D = 2 - alpha."""
    z0 = disc_points(n, radius, seed)
    b0 = K.newton_basin(z0)
    rng = SplitMix64(seed + 1)
    fr = []
    for e in eps_list:
        dz = e * np.exp(2j * math.pi * rng.uniform_array(n))
        b1 = K.newton_basin(z0 + dz)
        sel = (b0 >= 0) & (b1 >= 0)
        fr.append(float(np.mean(b0[sel] != b1[sel])))
    slope = float(np.polyfit(np.log(eps_list), np.log(fr), 1)[0])
    return {"eps": list(eps_list), "fraction": fr, "alpha": slope, "dimension": 2 - slope}


def boundary_map(n_pixels: int, half: float = 1.5, shift: float = 1.234567e-4):
    """Gebietskarte auf einem n x n Gitter in [-half, half]^2 (mit kleinem Versatz) und Randpixel (Nachbarn mit anderem Gebiet)."""
    g = np.linspace(-half, half, n_pixels) + shift
    lab = K.newton_basin(g[None, :] + 1j * g[:, None])
    diff_v = lab[:-1, :] != lab[1:, :]
    diff_h = lab[:, :-1] != lab[:, 1:]
    bd = np.zeros(lab.shape, bool)
    bd[:-1, :] |= diff_v
    bd[1:, :] |= diff_v
    bd[:, :-1] |= diff_h
    bd[:, 1:] |= diff_h
    return lab, bd, 2 * half / (n_pixels - 1)


def box_dimension(bd: np.ndarray, h: float, sizes=(2, 4, 8, 16, 32, 64, 128), use=slice(1, 6)) -> dict:
    """Box-Zählung der Randpixel: Zahl der Boxen (Seite s Pixel) mit Rand; Dimension = -Steigung von log N gegen log (s h) über die Boxgrößen `use`."""
    n = bd.shape[0]
    counts = []
    for s in sizes:
        k = (n // s) * s
        blk = bd[:k, :k].reshape(k // s, s, k // s, s).any(axis=(1, 3))
        counts.append(int(blk.sum()))
    xs = np.log([s * h for s in sizes[use]])
    ys = np.log(counts[use])
    return {"sizes": list(sizes), "counts": counts, "dimension": float(-np.polyfit(xs, ys, 1)[0]), "boundary_share": float(bd.mean())}


def image(center: complex, half: float, n: int = 240):
    """Gebietsbild eines Ausschnitts (n x n Punkte): Matrix mit Gebietsnummern 0, 1, 2 (-1: keine Konvergenz)."""
    xs = center.real + np.linspace(-half, half, n)
    ys = center.imag + np.linspace(-half, half, n)
    return xs, ys, K.newton_basin(xs[None, :] + 1j * ys[:, None])
