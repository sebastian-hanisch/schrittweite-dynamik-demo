"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sdy_constants as C

VIEW = 3.5


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_landscape(path_x, path_y):
    """Konturbild von log10(f + 1e-4) mit den Minima x·y = 1 (Hyperbel) und dem GD-Pfad (Punkte außerhalb des Bildes werden weggelassen)."""
    g = np.linspace(-VIEW, VIEW, 141)
    gx, gy = np.meshgrid(g, g)
    z = np.log10(0.5 * (gx * gy - 1) ** 2 + 1e-4)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=g, y=g, z=z, colorscale="Blues", showscale=False, contours=dict(showlines=False), hoverinfo="skip", opacity=0.8, name="log₁₀ f"))
    t = np.linspace(1 / VIEW, VIEW, 120)
    for sign in (1, -1):
        fig.add_trace(go.Scatter(x=sign * t, y=sign / t, mode="lines", line=dict(color=C.COLOR_OK, width=2, dash="dash"), name="Minima x·y = 1" if sign == 1 else None,
                                 showlegend=sign == 1, hoverinfo="skip"))
    inside = (np.abs(path_x) <= VIEW) & (np.abs(path_y) <= VIEW)
    fig.add_trace(go.Scatter(x=path_x[inside], y=path_y[inside], mode="lines+markers", line=dict(color=C.COLOR_PATH, width=2), marker=dict(size=5), name="GD-Pfad",
                             hovertemplate="(%{x:.3f}; %{y:.3f})<extra></extra>"))
    fig.add_trace(go.Scatter(x=[path_x[0]], y=[path_y[0]], mode="markers", marker=dict(size=11, color="black", symbol="x"), name="Start"))
    fig.update_xaxes(title_text="x", range=[-VIEW, VIEW])
    fig.update_yaxes(title_text="y", range=[-VIEW, VIEW])
    return _base(fig, 420)


def build_counts(cl):
    """Zählung der 900 Starts: konvergiert, divergiert, beschränkt ohne Konvergenz."""
    fig = go.Figure(go.Bar(x=["konvergiert", "divergiert", "beschränkt ohne Konvergenz"], y=[cl["converged"], cl["diverged"], cl["bounded"]],
                           marker_color=[C.COLOR_OK, C.COLOR_PATH, C.COLOR_LINE], hovertemplate="%{x}: %{y} Starts<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Zahl der Starts")
    return _base(fig, 320)


def build_sharp_hist(cl):
    """Histogramm der Schärfen der erreichten Minima mit der Grenze 2/η und der kleinstmöglichen Schärfe 2."""
    fig = go.Figure()
    sharp = cl["sharp"]
    if len(sharp):
        fig.add_trace(go.Histogram(x=sharp, nbinsx=25, marker_color=C.COLOR_OK, name="erreichte Minima"))
    fig.add_vline(x=cl["limit"], line=dict(color=C.COLOR_PATH, dash="dash"), annotation_text=f"2/η = {cl['limit']:.2f}".replace(".", ","), annotation_position="top")
    fig.add_vline(x=2.0, line=dict(color="grey", dash="dot"), annotation_text="kleinstmögliche Schärfe 2", annotation_position="bottom right")
    fig.update_xaxes(title_text="Schärfe S = x² + y² des erreichten Minimums", range=[1.8, max(2.2, min(30.0, cl["limit"] * 1.05))])
    fig.update_yaxes(title_text="Zahl der Starts")
    return _base(fig, 320)


def build_bifurcation(ex, tx, etas, lam, eta, thresholds):
    """Bifurkationsdiagramm der Diagonalabbildung (oben) und Lyapunov-Exponent (unten) mit den Verdopplungsschwellen und der gewählten Schrittweite."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    keep = np.abs(tx) < 2.5
    fig.add_trace(go.Scattergl(x=ex[keep], y=tx[keep], mode="markers", marker=dict(size=1.5, color=C.COLOR_LINE), name="t nach dem Einschwingen", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=etas, y=np.clip(lam, -1.0, 1.0), mode="lines", line=dict(color=C.COLOR_PATH, width=2), name="Lyapunov-Exponent", hovertemplate="η = %{x:.3f}: %{y:.3f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color="grey", dash="dot"), row=2, col=1)
    for th in thresholds:
        fig.add_vline(x=th, line=dict(color="rgba(150,150,150,0.8)", dash="dot"))
    fig.add_vline(x=eta, line=dict(color="black", width=2), annotation_text=f"η = {eta:.2f}".replace(".", ","), annotation_position="top")
    fig.update_xaxes(title_text="Schrittweite η", row=2, col=1)
    fig.update_yaxes(title_text="t", row=1, col=1, range=[-2.2, 2.2])
    fig.update_yaxes(title_text="λ", row=2, col=1, range=[-1.05, 0.7])
    return _base(fig, 520)


def build_diag_orbit(t):
    """Die ersten Werte der Bahn t_k der Diagonalabbildung."""
    fig = go.Figure(go.Scatter(x=list(range(len(t))), y=t, mode="lines+markers", line=dict(color=C.COLOR_LINE, width=2), marker=dict(size=5), name="t_k",
                               hovertemplate="k = %{x}: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Schritt k")
    fig.update_yaxes(title_text="t_k", range=[-2.2, 2.2])
    return _base(fig, 300)


def build_efs(series):
    """η·λ_max(Hesse) entlang einer beschränkten Bahn mit der Linie 2."""
    fig = go.Figure(go.Scatter(x=list(range(len(series))), y=series, mode="lines+markers", line=dict(color=C.COLOR_OK, width=2), marker=dict(size=5), name="η·λ_max",
                               hovertemplate="Schritt %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=2.0, line=dict(color=C.COLOR_PATH, dash="dash"), annotation_text="2", annotation_position="top left")
    fig.update_xaxes(title_text="Schritt")
    fig.update_yaxes(title_text="η · λ_max(Hesse)")
    return _base(fig, 300)


def build_newton(xs, ys, im):
    """Gebietsbild: Farbe = Wurzel, gegen die Newton konvergiert (grau: keine Konvergenz)."""
    z = np.where(im < 0, 3, im).astype(float)
    colors = list(C.COLOR_BASINS) + ["#bbbbbb"]
    scale = []
    for i, col in enumerate(colors):
        scale += [[i / 4, col], [(i + 1) / 4, col]]
    fig = go.Figure(go.Heatmap(x=xs, y=ys, z=z, zmin=0, zmax=4, colorscale=scale, showscale=False, hovertemplate="z = %{x:.3f} + %{y:.3f}i<extra></extra>"))
    fig.update_xaxes(title_text="Re z")
    fig.update_yaxes(title_text="Im z")
    return _base(fig, 440)
