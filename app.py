"""Schrittweite als Dynamik: Gradientenabstieg jenseits von 2/η, Periodenverdopplung und Newton-Fraktal - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der Linie „Dynamische Systeme“ (Kante: fix zu Stück 1). Aufbau nach DEMO-PLAYBOOK §2: Schnellstart, Schrittweite und Zufalls-Seed in der Seitenleiste, die Regler
der Experimente (Start, Ausschnitt) stehen im Hauptbereich beim Experiment, Kernfrage mit Verdikt vor dem Expander.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import sdy_algorithm as A
import sdy_constants as C
import sdy_core as K
import sdy_evaluation as ev
from sdy_layout import dunkles_design, kopf_css
from sdy_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from sdy_visualization import build_bifurcation, build_counts, build_diag_orbit, build_efs, build_landscape, build_newton, build_sharp_hist

st.set_page_config(page_title="Schrittweite als Dynamik – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _step(x, y, eta):
    return A.step_report(x, y, eta)


@st.cache_data(show_spinner=False)
def _path(x, y, eta):
    return K.gd_path(x, y, eta, C.PATH_STEPS)


@st.cache_data(show_spinner=False)
def _sharp(eta):
    return A.sharpness_counts(eta)


@st.cache_data(show_spinner=False)
def _bifurcation():
    etas = np.linspace(C.DIAG_ETA_LO, C.DIAG_ETA_HI, C.DIAG_ETA_N)
    ex, tx, lam = K.bifurcation(etas)
    return etas, ex, tx, lam


@st.cache_data(show_spinner=False)
def _thresholds():
    return K.doubling_thresholds()


@st.cache_data(show_spinner=False)
def _onset():
    rows = A.diag_grid()
    return next((r["eta"] for r in rows if r["kind"] == "chaos"), None)


@st.cache_data(show_spinner=False)
def _diag(eta):
    return A.diag_report(eta)


@st.cache_data(show_spinner=False)
def _efs(eta):
    return A.efs_report(eta)


@st.cache_data(show_spinner=False)
def _newton(seed):
    return A.newton_report(seed)


@st.cache_data(show_spinner=False)
def _image(cre, cim, half):
    return A.newton_image(complex(cre, cim), half)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in sdy_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📐 Schrittweite als Dynamik: Gradientenabstieg jenseits von 2/η, Periodenverdopplung und Newton-Fraktal")
st.markdown(
    """
Auf einer Quadratik konvergiert der **Gradientenabstieg** genau dann, wenn die Schrittweite η unter 2/λ ([Stück 1](https://github.com/sebastian-hanisch/fixpunkt-iteration-demo), die Schwelle steht scharf in
[gradient-descent-demo](https://github.com/sebastian-hanisch/gradient-descent-demo)); darüber wächst die Iteration. Bei einer **nichtlinearen** Funktion bleibt die Bahn jenseits der Schwelle oft beschränkt (im Spielmodell als **Zyklus** oder
chaotisch). Die Demo misst das an einem Spielmodell (von Hand gewählt, keine Quelle): f(x, y) = ½(x·y − 1)². Dazu das **Newton-Verfahren** für z³ = 1, dessen Einzugsgebiete eine fraktale Grenze haben.

Weiter unten: fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Schrittweite als Dynamik", expanded=True):
    st.markdown(
        """
1. **Das Modell.** f(x, y) = ½(x·y − 1)² hat als Minima die Kurve x·y = 1. Die **Schärfe** eines Minimums ist der größte Eigenwert der Hesse-Matrix dort, S = x² + y² (mindestens 2, bei x = y = 1).
2. **Der Schritt.** GD rechnet (x, y) → (x − η·y·c, y − η·x·c) mit c = x·y − 1. Dabei wird die Differenz D = x² − y² mit dem Faktor 1 − η²c² multipliziert: sie schrumpft für |ηc| < 1 und im Mittel stärker bei größerem η. Im Mittel läuft GD deshalb zu **flacheren** Minima.
3. **Stabilität.** Ein Minimum mit Schärfe S ist für GD stabil genau dann, wenn η·S < 2. Für η > 1 ist 2/η < 2: **kein Minimum ist mehr stabil**; bei η = 1 ist (1; 1) nur grenzstabil.
4. **Was dann geschieht.** Auf der Diagonalen x = y ist GD die Abbildung t → (1 + η)t − ηt³: Periodenverdopplung bei η = 1 und η = √5 − 1, dann Chaos.
5. **Newton** ist ein Fixpunktverfahren in der komplexen Ebene; jede der drei Wurzeln von z³ = 1 hat ein Einzugsgebiet, die Grenze dazwischen ist ein Fraktal.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Schrittweite**")
    eta = st.slider(
        "Schrittweite η", C.ETA_MIN, C.ETA_MAX, key="eta", step=C.ETA_STEP, format="%.2f",
        help="Gilt für die Experimente 1 bis 4. Voreinstellung 0,3 (keine Zweierpotenz); die Schwelle 2/S der Stabilität eines Minimums mit Schärfe S wandert mit η.",
    )
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt die Zufallspunkte des Newton-Experiments fest (eigener Generator SplitMix64).")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)
    st.caption("Der Start (x, y) und der Ausschnitt des Fraktals stehen im Hauptbereich beim Experiment.")

# --- Experiment 1: ein Schritt und die Landschaft -----------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: ein Schritt und die Landschaft")
c1, c2 = st.columns(2)
x0 = c1.slider("Start x", C.START_MIN, C.START_MAX, key="x", step=0.01, format="%.2f")
y0 = c2.slider("Start y", C.START_MIN, C.START_MAX, key="y", step=0.01, format="%.2f")
rep = _step(float(x0), float(y0), float(eta))
m1, m2, m3, m4 = st.columns(4)
m1.metric("c = x·y − 1", ev.de(rep["c"], 3))
m2.metric("x′", ev.de(rep["x1"], 3))
m3.metric("y′", ev.de(rep["y1"], 3))
m4.metric("Faktor 1 − η²c² für D", ev.de(rep["factor"], 4))
px, py = _path(float(x0), float(y0), float(eta))
st.plotly_chart(build_landscape(px, py), width="stretch", key="landscape_chart")
st.caption(f"Linien: Niveaulinien von log₁₀ f; gestrichelt die Kurve x·y = 1 der Minima; rot der GD-Pfad der ersten {C.PATH_STEPS} Schritte.")

st.markdown("## ❓ Kernfrage: Wohin läuft GD, und was geschieht jenseits von 2/η?")
kind, text = ev.verdict_step(rep)
getattr(st, kind)(text)
st.markdown(
    "In allen gezählten Starts sind die erreichten Minima flacher als 2/η (**Experiment 2**); für η > 1 gibt es kein stabiles Minimum, und die beschränkten Bahnen laufen in **Zyklen** oder werden chaotisch (**Experimente 3 und 4**); bei η = 1 laufen sie langsam gegen ein Minimum. "
    "Das Newton-Verfahren zeigt dasselbe Fixpunktdenken in der komplexen Ebene (**Experiment 5**)."
)

st.markdown("---")

# --- Experiment 2: Schärfe gegen 2/eta ---------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Schärfe der erreichten Minima gegen 2/η")
cl = _sharp(float(eta))
g1, g2 = st.columns(2)
g1.plotly_chart(build_counts(cl), width="stretch", key="counts_chart")
g2.plotly_chart(build_sharp_hist(cl), width="stretch", key="sharp_chart")
st.caption(f"{cl['n']} Starts im Quadrat [−2,5; 2,5]² (Raster 30 × 30 mit Versatz), {C.GRID_STEPS} Schritte; „konvergiert“ heißt Verlust unter 10⁻¹².")
kind, text = ev.verdict_sharp(cl)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3: die Diagonale ---------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wenn kein Minimum stabil ist – die Diagonale x = y")
etas, ex, tx, lam = _bifurcation()
th = _thresholds()
drep = _diag(float(eta))
st.plotly_chart(build_bifurcation(ex, tx, etas, lam, float(eta), th), width="stretch", key="bifurcation_chart")
st.plotly_chart(build_diag_orbit(drep["orbit"]), width="stretch", key="diag_orbit_chart")
st.caption("Oben die Werte t der Diagonalabbildung nach dem Einschwingen, unten ihr Lyapunov-Exponent (auf −1 bis 1 begrenzt); senkrecht die Verdopplungsschwellen und die gewählte Schrittweite. Unten die ersten 60 Werte ab t₀ = 1,2.")
kind, text = ev.verdict_diag(drep, th, _onset())
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4: eta * lambda_max ---------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: η·λ_max(Hesse) entlang der Bahn")
erep = _efs(float(eta))
if erep["n_bounded"]:
    st.plotly_chart(build_efs(erep["series"]), width="stretch", key="efs_chart")
trio = [_efs(e) for e in C.ETA_TRIO]
st.dataframe(
    pd.DataFrame(
        [
            {
                "η": ev.de(r["eta"], 2), "beschränkte Bahnen": r["n_bounded"], "Periode": ", ".join(str(p) if p else "keine" for p in r["periods"]), "Zeitmittel η·λ_max (Median)": ev.de(r["mean_median"], 3),
                "Wertebereich": f"{ev.de(r['range_min'], 3)} bis {ev.de(r['range_max'], 3)}", "Anteil der Zeit über 2": ev.de(r["above_median"], 3),
            }
            for r in trio if r["n_bounded"]
        ]
    ),
    hide_index=True, width="stretch",
)
st.caption("Tabelle: dieselben Größen für η = 1,01, 1,05 und 1,3, jeweils über die beschränkten Bahnen des 900er Rasters (exaktes Zeitmittel über eine Periode).")
kind, text = ev.verdict_efs(erep)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5: Newton-Fraktal --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Newton-Fraktal z³ = 1")
n1, n2, n3 = st.columns(3)
zoom = n1.select_slider("Ausschnitt (Halbbreite)", options=C.ZOOM_OPTIONS, key="zoom", format_func=lambda v: ev.de(v, 2))
cre = n2.slider("Mitte Re z", -2.0, 2.0, key="cre", step=0.01, format="%.2f")
cim = n3.slider("Mitte Im z", -2.0, 2.0, key="cim", step=0.01, format="%.2f")
xs, ys, im = _image(float(cre), float(cim), float(zoom))
st.plotly_chart(build_newton(xs, ys, im), width="stretch", key="newton_chart")
nrep = _newton(int(seed))
dim = ev.load_dimension()
kind, text = ev.verdict_newton(nrep, dim)
getattr(st, kind)(text)
st.caption("Farben: die drei Wurzeln von z³ = 1 (grau: keine Konvergenz in 60 Schritten). Die Dimension der Grenze ist vorgerechnet (`tools/vorrechnen.py dimension`); die Anteile werden hier mit dem Zufalls-Seed der Seitenleiste neu gemessen.")
if dim is not None:
    box = dim["box"]
    st.caption(
        f"Vorgerechnet: {dim['punkte']:,} Zufallspunkte, Zufalls-Seed {dim['seed']}; Box-Zählung auf einem Gitter {box['pixel']} × {box['pixel']} ({box['randpixel']:,} Randpixel)".replace(",", ".")
    )
else:
    st.info("Die vorgerechnete Dimensionsmessung (sdy_vorgerechnet_dimension.json) liegt nicht vor.")

sync_query_params({k: st.session_state[k] for k in ("seed_input", "eta", "x", "y", "zoom", "cre", "cim")})

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein selbst gewähltes Spielmodell** f = ½(x·y − 1)² | Keine der gelesenen Arbeiten wurde auf dieses Modell geprüft. Die Schärfe der Minima und die Zyklen jenseits von 2/η sind für dieses Modell gemessen (die Schwellen 1 und √5 − 1 zusätzlich exakt hergeleitet), nicht für Netze. | – |
| **Kein Netz, keine stochastischen Gradienten** | Cohen u. a. beschreiben für Netze, dass die Schärfe am Rand von 2/η bleibt; hier ist das nicht gemessen. Damian u. a. nennen die Dynamik „far from being chaotic“, bei mehreren instabilen Eigenwerten dagegen „significantly more chaotic“. Das Spielmodell hat auf der Diagonalen nur eine instabile Richtung. | [gradient-descent-demo](https://github.com/sebastian-hanisch/gradient-descent-demo) |
| **Ein Startquadrat, 900 Starts, 6000 Schritte** | Sehr langsam konvergierende Starts würden als „beschränkt“ gezählt (bei η = 0,99 gab es keinen, bei η = 1,0 sind es die 366 beschränkten); die Zählungen gelten für dieses Raster, nicht für alle Startpunkte. | – |
| **Chaos nur auf der Diagonalen, Gitter 0,0125** | Die Verdopplungsschwellen 1 und √5 − 1 sind exakt, die weiteren numerisch; die Abgrenzung von Chaos nutzt den Lyapunov-Exponenten über 0,003. | [Stück 2: logistische Abbildung](https://github.com/sebastian-hanisch/logistische-abbildung-demo) |
| **Newton: nur z³ = 1** | Die Dimension der Grenze ist mit zwei Verfahren gemessen, die um bis zu 0,03 voneinander abweichen; eine Literaturgegenprobe liegt nicht vor. | [newton-verfahren-demo](https://github.com/sebastian-hanisch/newton-verfahren-demo) |
| **Lernschrittweiten in Spielen** | Schwellen der Lernschrittweite bei Hedge sind dieselbe Art Aussage, aber ein anderes System. | [Stück 9](https://github.com/sebastian-hanisch/lernen-spiele-chaos-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $f(x,y) = \tfrac12 (xy - 1)^2$ mit dem Gradienten $\nabla f = (y\,c,\; x\,c)$, $c = xy - 1$. Die Minima sind die Kurve $xy = 1$; die Hesse-Matrix dort ist $\begin{pmatrix} y^2 & 1 \\ 1 & x^2 \end{pmatrix}$ mit den Eigenwerten $0$ und der Schärfe $S = x^2 + y^2 \ge 2$.

**Gradientenschritt.** $(x, y) \mapsto (x - \eta\, y\, c,\; y - \eta\, x\, c)$. Daraus folgt $x'^2 - y'^2 = (x^2 - y^2)\,(1 - \eta^2 c^2)$. Auf der Kurve $xy = 1$ gilt $S = \sqrt{4 + D^2}$ mit $D = x^2 - y^2$; $D$ schrumpft in jedem Schritt mit $|\eta c| < 1$ um den Faktor $1 - \eta^2 c^2$.

**Stabilität.** Die Jacobi-Matrix des Schritts im Minimum hat die Eigenwerte $1$ und $1 - \eta S$; das Minimum ist stabil genau für $\eta S < 2$, also für $S < 2/\eta$. Für $\eta > 1$ ist $2/\eta < 2 \le S$: kein stabiles Minimum; bei $\eta = 1$ ist $(1, 1)$ mit $S = 2 = 2/\eta$ nur grenzstabil.

**Diagonale.** Für $x = y = t$ ist der Schritt $g(t) = (1 + \eta)\,t - \eta\, t^3$ mit $g'(1) = 1 - 2\eta$. Der Zweierzyklus $(a, b)$ mit $a + b = \sqrt{1 + 3/\eta}$, $ab = 1/\eta$, $(a - b)^2 = 1 - 1/\eta$ existiert für $\eta > 1$ und hat den Multiplikator $9 - 2(1 + \eta)^2$: gleich $1$ bei $\eta = 1$ (Gabelung), gleich $-1$ bei $\eta = \sqrt 5 - 1 \approx 1{,}2361$ (zweite Verdopplung). Der Zyklus $(+a, -a)$ mit $a^2 = (2 + \eta)/\eta$ hat den Multiplikator $(5 + 2\eta)^2 > 1$ und ist immer instabil.

**Newton.** $N(z) = z - \dfrac{z^3 - 1}{3 z^2} = \dfrac{2 z^3 + 1}{3 z^2}$; nahe einer Wurzel gilt $e' \approx e^2$. Mit $\omega = e^{2\pi i/3}$ ist $N(\omega z) = \omega N(z)$, daher haben die drei Einzugsgebiete in einer Kreisscheibe um $0$ gleiche Größe. Die Dimension der Grenze folgt aus dem Unsicherheitsexponenten $\alpha$ (Anteil der Punkte, die bei einer Störung $\varepsilon$ das Gebiet wechseln, wächst wie $\varepsilon^\alpha$) als $D = 2 - \alpha$ und aus der Box-Zählung.

**Quellen.** Cohen u. a., ICLR 2021 (im Wortlaut gelesen): Schärfe ist der größte Eigenwert der Hesse-Matrix; auf Quadratiken ist GD stabil genau dann, wenn die Schärfe unter $2/\eta$ liegt; bei Netzen bleibt sie am Rand von $2/\eta$. Damian, Nichani, Lee, ICLR 2023 (im Wortlaut gelesen). Die Messungen dieser Demo sind eigene.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
