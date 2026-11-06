import numpy as np
from matplotlib.ticker import FuncFormatter

COLORS = {
    "chik": "#1F5E8C",
    "dengue": "#B45A19",
    "zika": "#6750A4",
    "observed": "#D9D9D9",
    "observed_edge": "#4D4D4D",
    "grid": "#D8D8D8",
    "identity": "#8B8B8B",
}

def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson/PCHIP derivatives; shape preserving and dependency-free."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    d = np.zeros(n, dtype=float)
    if n == 2:
        d[:] = delta[0]
        return d
    same = (delta[:-1] * delta[1:]) > 0
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    d[1:-1][same] = (w1[same] + w2[same]) / (
        w1[same] / delta[:-1][same] + w2[same] / delta[1:][same]
    )
    d0 = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d0) != np.sign(delta[0]):
        d0 = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d0) > abs(3 * delta[0]):
        d0 = 3 * delta[0]
    dn = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(dn) != np.sign(delta[-1]):
        dn = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(dn) > abs(3 * delta[-1]):
        dn = 3 * delta[-1]
    d[0], d[-1] = d0, dn
    return d

def pchip(x: np.ndarray, y: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xnew = np.asarray(xnew, dtype=float)
    d = pchip_slopes(x, y)
    idx = np.clip(np.searchsorted(x, xnew) - 1, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (xnew - x[idx]) / h
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    out = h00 * y[idx] + h10 * h * d[idx] + h01 * y[idx + 1] + h11 * h * d[idx + 1]
    return np.clip(out, 0, None)

def style_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.7, pad=1.5)
    if grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)

def panel_letter(ax, letter, x=-0.18, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

def comma(x, pos):
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:g}"

def pseudo(x):
    return np.log10(np.asarray(x, dtype=float) + 1.0)
