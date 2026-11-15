"""Konstanten: Regler-Grenzen, feste Annahmen, Zufalls-Seeds, Farben und Presets."""

# --- Schrittweite und Startwert (Spielmodell f = 1/2 (x y - 1)^2) --------------------------------------------------------------------------
ETA_MIN, ETA_MAX, ETA_STEP = 0.05, 2.1, 0.01
START_MIN, START_MAX = -3.0, 3.0
DEFAULT_ETA = 0.3                      # keine Zweierpotenz als Voreinstellung; der Start (3; 0,1) konvergiert hier (bei 0,49 divergiert er)
DEFAULT_X, DEFAULT_Y = 3.0, 0.1        # Handbeispiel
PATH_STEPS = 40

# --- Raster der Starts (Experiment 2 und 4) -------------------------------------------------------------------------------------------------
GRID_N = 30
GRID_LO, GRID_HI = -2.5, 2.5
GRID_OFFSET = (0.0137, -0.0291)
GRID_STEPS = 6000
ETA_TRIO = (1.01, 1.05, 1.3)           # Experiment 4: Schrittweiten jenseits der Schwelle
ETA_NEIGHBOURS = (0.49, 0.51, 0.99, 1.01, 1.99, 2.01, 0.5, 1.0, 2.0)   # die 9 Schrittweiten der Zählung über Minima mit S >= 2/eta

# --- Diagonale (Experiment 3) ----------------------------------------------------------------------------------------------------------------
DIAG_ETA_LO, DIAG_ETA_HI, DIAG_ETA_N = 0.9, 1.75, 680
DIAG_GRID = (0.9012, 0.0125, 68)       # Gitter der Vormessung: 0,9012 + 0,0125 k (Zählung der Perioden und des Chaos)
DIAG_T0 = 1.2

# --- Newton-Fraktal (Experiment 5) ------------------------------------------------------------------------------------------------------------
SEED_MAX, DEFAULT_SEED = 999999, 2025
NEWTON_POINTS = 200_000                # Zufallspunkte der Live-Messung der Anteile
NEWTON_RADIUS = 1.5
IMAGE_N = 240
ZOOM_OPTIONS = (1.5, 0.75, 0.3, 0.1, 0.03)
DEFAULT_ZOOM = 1.5
DEFAULT_CENTER = (0.0, 0.0)
DIMENSION_SEED = 7001

# --- Farben ------------------------------------------------------------------------------------------------------------------------------------
COLOR_PATH = "#c0504d"
COLOR_LINE = "#4c78a8"
COLOR_OK = "#2F6B65"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_GREY = "rgba(150,150,150,0.55)"
COLOR_BASINS = ("#4c78a8", "#e09f3e", "#2F6B65")

# --- Presets (jeder Regler steht in jedem Preset; Schlüssel wie in sdy_presets.SETTING_SPECS) ---------------------------------------------
DEFAULTS = {"seed": DEFAULT_SEED, "eta": DEFAULT_ETA, "x": DEFAULT_X, "y": DEFAULT_Y, "zoom": DEFAULT_ZOOM, "cre": DEFAULT_CENTER[0], "cim": DEFAULT_CENTER[1]}
PRESETS = {
    "Handbeispiel (3; 0,1)": dict(DEFAULTS),
    "Schärfe gegen 2/η": {**DEFAULTS, "eta": 0.8, "x": 1.37, "y": -0.52},
    "Jenseits der Schwelle (η = 1,01)": {**DEFAULTS, "eta": 1.01},
    "Periode 16 (η = 1,3)": {**DEFAULTS, "eta": 1.3},
    "Chaos (η = 1,5)": {**DEFAULTS, "eta": 1.5},
    "Newton-Fraktal": {**DEFAULTS, "zoom": 0.3, "cre": 0.35, "cim": 0.2},
}
PRESET_HELP = {
    "Handbeispiel (3; 0,1)": "Start (3; 0,1), η = 0,3: ein Gradientenschritt von Hand, die Differenz x² − y² schrumpft um den Faktor 1 − η²c².",
    "Schärfe gegen 2/η": "η = 0,8: 900 Starts, die erreichten Minima sind alle flacher als 2/η = 2,5.",
    "Jenseits der Schwelle (η = 1,01)": "Für η > 1 gibt es kein stabiles Minimum mehr: kein Start konvergiert, die beschränkten Bahnen laufen in einen Zweierzyklus.",
    "Periode 16 (η = 1,3)": "Auf der Diagonalen x = y: nach drei weiteren Verdopplungen ein Zyklus der Periode 16.",
    "Chaos (η = 1,5)": "Die Diagonalabbildung hat hier einen positiven Lyapunov-Exponenten; in der Ebene entkommen die meisten Bahnen.",
    "Newton-Fraktal": "Newton-Verfahren für z³ = 1, ein Ausschnitt nahe der fraktalen Grenze der drei Einzugsgebiete.",
}
