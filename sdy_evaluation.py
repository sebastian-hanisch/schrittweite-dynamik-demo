"""Verdikte (reine Funktionen (Art, Text)), Zahlenformate und Laden der vorgerechneten Datei."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import sdy_constants as C

DIMENSION_FILE = Path(__file__).resolve().parent / "sdy_vorgerechnet_dimension.json"


@dataclass(frozen=True)
class Settings:
    seed: int = C.DEFAULTS["seed"]
    eta: float = C.DEFAULTS["eta"]
    x: float = C.DEFAULTS["x"]
    y: float = C.DEFAULTS["y"]
    zoom: float = C.DEFAULTS["zoom"]
    cre: float = C.DEFAULTS["cre"]
    cim: float = C.DEFAULTS["cim"]


@lru_cache(maxsize=None)
def load_dimension() -> dict | None:
    """Vorgerechnete Dimensionsmessung oder None, wenn die Datei fehlt (die App zeigt dann einen Hinweis)."""
    try:
        return json.loads(DIMENSION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def de(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    if x == float("inf"):
        return "∞"
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def groups(n: int) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{n:,}".replace(",", ".")


def percent(part: int, whole: int) -> str:
    return f"{de(100.0 * part / whole, 1)} %" if whole else "–"


# ----------------------------------------------------------------------------------------------------------------------------- Verdikte
def is_edge(eta: float) -> bool:
    """η = 1: das Minimum (1; 1) hat S = 2 = 2/η und ist nur grenzstabil (der Regler erlaubt diesen Wert)."""
    return abs(eta - 1.0) < 1e-9


def verdict_step(r: dict) -> tuple[str, str]:
    """Experiment 1: ein Schritt, die Differenz D = x² − y² und das Ende der Bahn."""
    f = r["factor"]
    if abs(r["c"]) < 1e-9:
        return "info", f"Der Start ({de(r['x'])}; {de(r['y'])}) liegt auf der Kurve x·y = 1 (c = 0): jeder Punkt dort ist ein Fixpunkt, die Bahn bleibt stehen; seine Schärfe ist S = {de(r['x'] ** 2 + r['y'] ** 2, 3)}."
    if f >= 1 - 1e-12:
        step = "D bleibt in diesem Schritt gleich (η·c = 0)"
    elif f >= 0:
        step = f"D = x² − y² schrumpft in diesem Schritt um den Faktor 1 − η²c² = {de(f, 4)} (von {de(r['D'], 4)} auf {de(r['D1'], 4)})"
    else:
        step = f"D = x² − y² wechselt in diesem Schritt das Vorzeichen (Faktor 1 − η²c² = {de(f, 4)} < 0, von {de(r['D'], 4)} auf {de(r['D1'], 4)})"
    if r["status"] == "converged":
        return "success", (f"{step}. Die Bahn konvergiert nach {r['steps']} Schritten in das Minimum ({de(r['xe'], 4)}; {de(r['ye'], 4)}); dort gilt S = √(4 + D²) = "
                           f"{de(r['sharp'], 4)} (gemessen x² + y²) und liegt unter 2/η = {de(r['limit'], 4)}.")
    if r["status"] == "diverged":
        return "warning", f"{step}. Die Bahn divergiert (η = {de(r['eta'])}; für ein Minimum mit Schärfe S wäre η < 2/S nötig, kleinstmöglich ist S = 2)."
    per = r["period"]
    if per:
        ende = f"sie läuft in einen Zyklus der Periode {per}."
    elif is_edge(r["eta"]):
        ende = f"bis zu Periode 128 ist kein Zyklus erkennbar (beschränkt nach 6000 Schritten); bei η = 1 ist das Minimum nur grenzstabil, in der Zählung (Experiment 2) laufen solche Bahnen langsam gegen ein Minimum."
    else:
        ende = "bis zu Periode 128 ist kein Zyklus erkennbar (beschränkt nach 6000 Schritten)."
    return "info", f"{step}. Die Bahn konvergiert nicht und divergiert nicht: " + ende


def verdict_sharp(cl: dict) -> tuple[str, str]:
    """Experiment 2: Zählung der 900 Starts und größte Schärfe der erreichten Minima gegen 2/η."""
    n, k, eta = cl["n"], cl["converged"], cl["eta"]
    head = (f"Von {n} Starts konvergieren {k}, {cl['diverged']} divergieren, {cl['bounded']} bleiben beschränkt ohne zu konvergieren (6000 Schritte, Verlust unter 10⁻¹²)")
    if cl["too_sharp"]:
        return "warning", head + f"; {cl['too_sharp']} der erreichten Minima haben Schärfe S ≥ 2/η = {de(cl['limit'], 4)}."
    if k == 0 and is_edge(eta):
        slow = (" Die beschränkten Bahnen sind aber keine Zyklen: sie laufen langsam gegen das Minimum (1; 1) bzw. (−1; −1), der Verlust sinkt weiter "
                "(in einer Stichprobe bis 3·10⁶ Schritte gemessen)." if cl["bounded"] else "")
        return "info", (head + f". Bei η = 1 ist 2/η = 2 und damit gleich der kleinstmöglichen Schärfe der Minima: (1; 1) ist nur grenzstabil, "
                               "kein Start erreicht die Schwelle für „konvergiert“." + slow)
    if k == 0:
        return "success" if eta > 1 else "info", (head + f". Bei η = {de(eta)} ist 2/η = {de(cl['limit'], 4)} kleiner als 2, die kleinstmögliche Schärfe der Minima ist 2: "
                                                  "es gibt kein stabiles Minimum, also konvergiert kein Start." if eta > 1 else head + ".")
    return "success", head + f". Die größte Schärfe der erreichten Minima ist {de(cl['sharp_max'], 4)}, unter 2/η = {de(cl['limit'], 4)}; kein erreichtes Minimum hat S ≥ 2/η."


def verdict_diag(rep: dict, thresholds, onset: float | None) -> tuple[str, str]:
    """Experiment 3: Art der Bahn der Diagonalabbildung bei η, mit den Verdopplungsschwellen."""
    eta, per, lam = rep["eta"], rep["period"], rep["lyapunov"]
    th = ", ".join(de(t, 4) if i else "1" for i, t in enumerate(thresholds))
    tail = f" Verdopplungsschwellen bei {th}" + (f"; Chaos auf dem Gitter ab η = {de(onset, 4)}." if onset is not None else ".")
    if per is None:
        return "warning", f"Bei η = {de(eta)} divergiert die Bahn der Diagonalabbildung (t wächst über alle Grenzen)." + tail
    if per > 0:
        return "success", f"Bei η = {de(eta)} läuft die Bahn in einen Zyklus der Periode {per} (Lyapunov-Exponent {de(lam, 4)}); kein Chaos." + tail
    if lam > 0.003:
        return "success", f"Bei η = {de(eta)} hat die Bahn keine Periode bis 128 und einen Lyapunov-Exponenten von +{de(lam, 3)} (über 0,003): chaotisch." + tail
    if is_edge(eta):
        return "info", (f"Bei η = 1 ist der Fixpunkt t = 1 der Diagonalabbildung nur grenzstabil (Multiplikator g′(1) = −1): die Bahn hat keine Periode bis 128 und läuft sehr langsam hinein, "
                        f"Lyapunov-Exponent {de(lam, 4)} (höchstens 0,003): nicht als Chaos eingestuft." + tail)
    return "info", f"Bei η = {de(eta)} hat die Bahn keine Periode bis 128, aber einen Lyapunov-Exponenten von nur {de(lam, 4)} (höchstens 0,003): hier nicht als Chaos eingestuft." + tail


def verdict_efs(r: dict) -> tuple[str, str]:
    """Experiment 4: η·λ_max(Hesse) entlang der beschränkten Bahnen (Spielmodell, keine Aussage über Netze)."""
    if r["n_bounded"] == 0:
        return "info", (f"Bei η = {de(r['eta'])} gibt es im Raster keine beschränkte Bahn ohne Konvergenz ({r['converged']} konvergieren, {r['diverged']} divergieren); "
                        "η·λ_max ist nur entlang beschränkter Bahnen definiert, die nicht im Minimum stehen.")
    per = r["periods"]
    kinds = ", ".join(f"Periode {p} ({c} Bahnen)" if p else f"ohne Periode bis 128 ({c} Bahnen)" for p, c in sorted(per.items()))
    how = "exaktes Zeitmittel über eine Periode" if r["exact"] else "Mittel über 4000 Schritte, bei Bahnen ohne Periode eine Näherung"
    chaos = f" Lyapunov-Exponent im Median {de(r['lyapunov_median'], 3)}, davon positiv (über 0,003): {r['n_chaotic']} von {r['n_bounded']}." if r["n_chaotic"] else ""
    if is_edge(r["eta"]):
        chaos += " Bei η = 1 laufen diese Bahnen langsam gegen ein Minimum (Experiment 2): „ohne Periode“ heißt hier weder Zyklus noch Chaos."
    return "info", (f"Bei η = {de(r['eta'])}: {r['n_bounded']} beschränkte Bahnen, {kinds}. η·λ_max(Hesse) im Zeitmittel (Median über die Bahnen, {how}) {de(r['mean_median'], 3)}, "
                    f"Wertebereich {de(r['range_min'], 3)} bis {de(r['range_max'], 3)}, Anteil der Zeit über 2 im Median {de(r['above_median'], 3)}.{chaos} "
                    "Das gilt in diesem Spielmodell; ob Netze sich so verhalten, ist hier nicht gemessen.")


def verdict_newton(rep: dict, dim: dict | None) -> tuple[str, str]:
    """Experiment 5: Anteile der Einzugsgebiete in der Scheibe und Dimension der Grenze (zwei Verfahren, vorgerechnet)."""
    a = rep["shares"]
    text = (f"Anteile der drei Einzugsgebiete in der Scheibe |z| ≤ {de(rep['radius'], 1)} ({groups(rep['n'])} Zufallspunkte): "
            + ", ".join(de(100 * s, 1) + " %" for s in a) + " (Symmetrie N(ωz) = ωN(z): je 1/3).")
    if dim is None:
        return "info", text + " Die vorgerechnete Dimensionsmessung liegt nicht vor."
    u, bw, bf = dim["unsicherheit"]["dimension"], dim["box"]["dimension_4_bis_64"], dim["box"]["dimension_2_bis_16"]
    lo, hi = min(u, bw, bf), max(u, bw, bf)
    return "info", text + f" Dimension der Grenze (vorgerechnet, zwei Verfahren): {de(lo, 2)} bis {de(hi, 2)} (Unsicherheitsexponent {de(u, 3)}, Box-Zählung {de(bw, 3)} und {de(bf, 3)}); keine Literaturgegenprobe."
