"""V4 trend+CoV Figure 4 with ten-seed mean point predictions.

The a-h panel geometry comes from the exact 2026-09-30 submission renderer.
Zika panel c displays 2016 monthly H7 targets from continuous replay;
the complete August 2015-December 2016 table remains in the source package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


from .style import apply_style
from . import figure4_layout as source
SUMMARY_DIR = None
ZIKA_C = None
OUT = None


def _load_reference():
    return source


def _tsv(name: str) -> pd.DataFrame:
    return pd.read_csv(SUMMARY_DIR / name, sep="\t")


def load_panels() -> dict[str, pd.DataFrame]:
    national = _tsv("Figure4_bounded_adapter_national_summary.tsv")
    department = _tsv("Figure4_bounded_adapter_department_month_summary.tsv")
    peak = _tsv("Figure4_bounded_adapter_peak_summary.tsv")
    total = _tsv("Figure4_bounded_adapter_cumulative_summary.tsv")
    strict_all = pd.read_csv(ZIKA_C, sep="\t")
    if len(strict_all) != 17 or int(strict_all.n_days.sum()) != 489:
        raise ValueError("The full continuous Zika source must retain 17 months and 489 H7 target days")
    strict = strict_all[strict_all.month.astype(str).between("2016-01", "2016-12")].copy()
    panels = {
        "a": national[national.disease == "chik"].copy(),
        "b": national[national.disease == "dengue"].copy(),
        "c": strict.copy(),
        "d": department[department.department == "Antioquia"].copy(),
        "e": department[department.department == "Cesar"].copy(),
        "f": department[department.department == "Arauca"].copy(),
        "g": peak.copy(),
        "h": total.copy(),
    }
    for key in "abcdef":
        frame = panels[key]
        frame["month"] = pd.to_datetime(frame["month"])
        panels[key] = frame.sort_values("month").reset_index(drop=True)
        required = {"month", "observed", "mean", "lower_95", "upper_95", "n_seed"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Panel {key} lacks required plotting columns")
        if not np.isfinite(frame[["observed", "mean", "lower_95", "upper_95"]].to_numpy()).all():
            raise ValueError(f"Panel {key} contains non-finite plotted values")
        if not (frame["n_seed"] == 10).all():
            raise ValueError(f"Panel {key} does not have all ten locked seeds")
    if {key: len(panels[key]) for key in "abcdef"} != dict(a=12, b=12, c=12, d=12, e=12, f=12):
        raise ValueError("Unexpected Figure 4 monthly panel coverage")
    if list(panels["c"].month.dt.strftime("%Y-%m")) != list(pd.period_range("2016-01", "2016-12", freq="M").astype(str)):
        raise ValueError("The displayed Zika monthly calendar must have January-December 2016")
    if int(panels["c"].n_days.sum()) != 365:
        raise ValueError("Displayed Zika H7 coverage must have 365 distinct target days")
    if int(panels["c"].n_seed.min()) != 10:
        raise ValueError("Continuous Zika panel requires all ten locked seeds")
    if len(panels["g"]) != 33 or len(panels["h"]) != 33:
        raise ValueError("All 33 departments are required")
    return panels


NAME_MAP = {
    "Atlantico": "Atlántico", "BogotaDC": "Bogotá D.C.", "Bolivar": "Bolívar",
    "Boyaca": "Boyacá", "Caqueta": "Caquetá", "Choco": "Chocó",
    "Cordoba": "Córdoba", "Guainia": "Guainía", "LaGuajira": "La Guajira",
    "Narino": "Nariño", "NortedeSantander": "Norte de Santander",
    "Quindio": "Quindío", "SanAndresyProvidencia": "San Andrés y\nProvidencia",
    "ValledelCauca": "Valle del Cauca", "Vaupes": "Vaupés",
}


def pretty_name(name: str) -> str:
    return NAME_MAP.get(str(name), str(name))


def rank_name(name: str) -> str:
    return pretty_name(name)


def _style_setup():
    apply_style((7.3, 6.3, 6.0))
    # The supplied PDF is exactly 7.08 x 6.45 inches.  No tight cropping.
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7.0,
        "axes.titlesize": 8.4, "axes.titleweight": "bold", "axes.titlelocation": "center",
        "axes.labelsize": 7.3, "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": None,
    })


def _legend_axis(ax, *, bars: bool):
    ax.axis("off")
    obs_x, pred_x = 0.095, 0.285
    ax.add_patch(Rectangle((obs_x, 0.30 if not bars else 0.26), 0.024,
                           0.40 if not bars else 0.48, transform=ax.transAxes,
                           facecolor="#D9D9D9", edgecolor="#4D4D4D", linewidth=0.55))
    ax.text(obs_x + 0.031, 0.50, "Observed monthly cases", transform=ax.transAxes,
            va="center", fontsize=5.0)
    if bars:
        ax.add_patch(Rectangle((pred_x, 0.26), 0.024, 0.48, transform=ax.transAxes,
                               facecolor="#B45A19", edgecolor="#5F2C0C", linewidth=0.5))
        ax.text(pred_x + 0.031, 0.50, "Predicted monthly cases", transform=ax.transAxes,
                va="center", fontsize=5.0)
    else:
        ax.plot([pred_x, pred_x + 0.026], [0.50, 0.50], transform=ax.transAxes,
                color="#1F5E8C", marker="o", markersize=2.3, linewidth=1.35)
        ax.text(pred_x + 0.033, 0.50, "Predicted cases", transform=ax.transAxes,
                va="center", fontsize=5.0)


def _national_panel(source, ax, frame: pd.DataFrame, key: str, title: str):
    color = source.COLORS[{"a": "chik", "b": "dengue", "c": "zika"}[key]]
    x = mdates.date2num((frame.month if key != "c" else pd.to_datetime(frame.plot_date)).to_numpy())
    fine = np.linspace(x.min(), x.max(), 420 if key == "c" else 260)
    centre = source.pchip(x, frame["mean"].to_numpy(), fine)
    lower = source.pchip(x, frame["lower_95"].to_numpy(), fine)
    upper = source.pchip(x, frame["upper_95"].to_numpy(), fine)
    ax.bar(x, frame.observed, width=18 if key != "c" else 14,
           facecolor="#D9D9D9", edgecolor="#4D4D4D", linewidth=0.55, zorder=2)
    ax.fill_between(fine, lower, upper, color=color, alpha=0.18,
                    linewidth=0, zorder=1)
    ax.plot(fine, centre, color=color, linewidth=1.7, zorder=3)
    ax.scatter(x, frame["mean"], s=9, color=color, edgecolors="none", zorder=4)
    ax.set_title(title, pad=6)
    if key == "c":
        ax.set_xlim(pd.Timestamp("2015-12-30"), pd.Timestamp("2016-12-31"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n2016"))
        # Exact H7 target-date ranges in the archived 3:1:1 design. Thin
        # unfilled spaces between bands are retrospective boundary replay.
        phase_bands = (
            ("Training", "2016-01-01", "2016-05-23", "#E1E5EA"),
            ("Validation", "2016-06-12", "2016-09-09", "#DFEAE8"),
            ("Test", "2016-09-29", "2016-12-27", "#E7DDF0"),
        )
        for label, start, end_exclusive, fill in phase_bands:
            left, right = mdates.date2num(pd.Timestamp(start)), mdates.date2num(pd.Timestamp(end_exclusive))
            ax.add_patch(Rectangle((left, 0.935), right-left, 0.06,
                                   transform=ax.get_xaxis_transform(),
                                   facecolor=fill, edgecolor="none", zorder=5))
            ax.text((left+right)/2, 0.965, label,
                    transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=5.0, color="#3B3B3B", zorder=6)
    else:
        ax.set_xlim(pd.Timestamp("2015-12-18"), pd.Timestamp("2016-12-17"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n2016"))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(source.comma))
    source.style_axis(ax)
    source.panel_letter(ax, key)
    if key == "a":
        ax.set_ylabel("Monthly cases")
    elif key == "c":
        ax.set_ylim(0, max(frame[["observed", "upper_95"]].to_numpy().max() * 1.14, 1))


def _department_panel(source, ax, frame: pd.DataFrame, key: str, title: str):
    x = np.arange(len(frame))
    width = 0.32
    ax.bar(x - width / 2, frame.observed, width, facecolor="#D9D9D9",
           edgecolor="#4D4D4D", linewidth=0.55, zorder=2)
    ax.bar(x + width / 2, frame["mean"], width, facecolor="#B45A19",
           edgecolor="#5F2C0C", linewidth=0.5, zorder=2)
    ax.errorbar(x + width / 2, frame["mean"],
                yerr=np.vstack([frame["mean"] - frame["lower_95"],
                                frame["upper_95"] - frame["mean"]]),
                fmt="none", ecolor="#5F2C0C", elinewidth=0.55, capsize=1.8, zorder=3)
    mae = np.abs(frame.observed.to_numpy() - frame["mean"].to_numpy()).mean()
    ax.set_title(title, pad=6)
    ax.text(0.98, 0.96, f"MAE = {mae:,.0f} cases", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.5))
    ax.set_xticks([0, 2, 4, 6, 8, 10],
                  ["Jan\n2016", "Mar\n2016", "May\n2016", "Jul\n2016", "Sep\n2016", "Nov\n2016"])
    ax.set_xlim(-0.55, 11.55)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(source.comma))
    source.style_axis(ax)
    source.panel_letter(ax, key)
    if key == "d":
        ax.set_ylabel("Monthly cases")


def _peak_panel(source, ax, frame: pd.DataFrame):
    gx = source.pseudo(frame["observed"])
    gy = source.pseudo(frame["mean"])
    maxv = float(max(gx.max(), gy.max()) + 0.04)
    ax.plot([0, maxv], [0, maxv], color="#8B8B8B", linewidth=0.9, zorder=1)
    ax.scatter(gx, gy, s=12, color="#CB7C56", edgecolors="#77422E", linewidths=0.35, zorder=3)
    ranked = frame.assign(absolute_log1p_peak_error=np.abs(
        np.log1p(frame["mean"].to_numpy()) - np.log1p(frame["observed"].to_numpy())))
    top_five = ranked.sort_values(["absolute_log1p_peak_error", "department"]).head(5)
    selected = set(top_five.department)
    expected = {"Antioquia", "Amazonas", "Santander", "ValledelCauca", "Atlantico"}
    if selected != expected:
        raise RuntimeError(f"Re-audit the five peak-error callout positions: {selected}")
    label_slots = {
        "ValledelCauca": (-4, -4, "right"),
        "Antioquia": (-8, -11, "right"),
        "Santander": (-4, -12, "right"),
        "Atlantico": (-12, 12, "right"),
        "Amazonas": (8, 1, "left"),
    }
    for row in frame.itertuples(index=False):
        if row.department not in selected:
            continue
        tx, ty, align = label_slots[row.department]
        ax.annotate(pretty_name(row.department),
                    (source.pseudo(row.observed), source.pseudo(row.mean)),
                    xytext=(tx, ty), textcoords="offset points", fontsize=5.8,
                    fontweight="bold",
                    ha=align, va="center", color="#3D3D3D",
                    arrowprops=dict(arrowstyle="-", color="#A0A0A0",
                                    linewidth=0.4, shrinkA=1, shrinkB=2))
    ax.text(0.98, 0.04, "n = 33 departments", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.0, color="#555555")
    ticks = np.array([0, 1, 3, 10, 30, 100, 300, 1000])
    ax.set_xticks(source.pseudo(ticks), [f"{v:,}" for v in ticks])
    ax.set_yticks(source.pseudo(ticks), [f"{v:,}" for v in ticks])
    ax.set_xlim(0, maxv)
    ax.set_ylim(0, maxv)
    ax.set_xlabel("Observed peak cases")
    ax.set_ylabel("Predicted peak cases")
    source.style_axis(ax)
    source.panel_letter(ax, "g", x=-0.18, y=1.06)


def _burden_panel(source, ax, frame: pd.DataFrame):
    frame = frame.sort_values("observed", ascending=False).reset_index(drop=True)
    x = np.arange(len(frame), dtype=float)
    # Give the final two long category labels enough horizontal separation.
    # This is a categorical display coordinate; every marker retains its label.
    if list(frame.department.tail(2)) == ["SanAndresyProvidencia", "Vaupes"]:
        x[-1] += float(os.environ.get("FIGURE4_H_LAST_EXTRA", "1.357"))
    observed = source.pseudo(frame["observed"])
    predicted = source.pseudo(frame["mean"])
    for xi, oy, py in zip(x, observed, predicted):
        ax.plot([xi, xi], [oy, py], color="#C8C8C8", linewidth=0.5, zorder=1)
    ax.scatter(x - 0.07, observed, marker="o", s=12, facecolor="#D9D9D9",
               edgecolor="#4D4D4D", linewidth=0.45, label="Observed", zorder=3)
    ax.scatter(x + 0.07, predicted, marker="^", s=17, facecolor="#B45A19",
               edgecolor="#5F2C0C", linewidth=0.4, label="Predicted", zorder=4)
    y_ticks = np.array([10, 30, 100, 300, 1000, 3000, 10000, 30000])
    ax.set_yticks(source.pseudo(y_ticks), [f"{v:,}" for v in y_ticks])
    ax.set_xticks(x, [rank_name(v) for v in frame.department],
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, float(x[-1]) + 0.7)
    ax.set_ylabel("Cumulative 7-day cases", labelpad=2)
    observed_variation = np.sum((observed - observed.mean()) ** 2)
    if observed_variation <= 0:
        raise ValueError("Figure 4h R² requires variation across observed departments")
    r2 = 1.0 - np.sum((predicted - observed) ** 2) / observed_variation
    ax.text(0.99, 0.96, f"R² = {r2:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 1.02), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)
    source.style_axis(ax)
    ax.tick_params(axis="x", labelsize=5.0, pad=2)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_rotation_mode("anchor")
        label.set_ha("right")
        label.set_va("top")
        label.set_multialignment("right")
        label.set_wrap(False)
    source.panel_letter(ax, "h", x=-0.045, y=1.06)


def build_figure(panels: dict[str, pd.DataFrame]):
    source = _load_reference()
    _style_setup()
    fig = plt.figure(figsize=(7.08, 6.45), constrained_layout=False)
    gs = fig.add_gridspec(5, 3, width_ratios=[1, 1, 1],
                          height_ratios=[0.12, 1.0, 0.10, 1.0, 1.40],
                          left=0.065, right=0.985, bottom=0.155, top=0.988,
                          wspace=0.34, hspace=0.56)
    legend_top = fig.add_subplot(gs[0, :])
    national = [fig.add_subplot(gs[1, i]) for i in range(3)]
    legend_middle = fig.add_subplot(gs[2, :])
    departments = [fig.add_subplot(gs[3, i]) for i in range(3)]
    axes = dict(zip("abcdefgh", national + departments + [fig.add_subplot(gs[4, 0]),
                                                          fig.add_subplot(gs[4, 1:3])]))
    _legend_axis(legend_top, bars=False)
    _legend_axis(legend_middle, bars=True)
    for key, title in zip("abc", ["Chikungunya", "Dengue", "Zika"]):
        _national_panel(source, axes[key], panels[key], key, title)
    for key, title in zip("def", ["Antioquia", "Cesar", "Arauca"]):
        _department_panel(source, axes[key], panels[key], key, title)
    _peak_panel(source, axes["g"], panels["g"])
    _burden_panel(source, axes["h"], panels["h"])
    return fig, axes


def main():
    global SUMMARY_DIR, ZIKA_C, OUT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    SUMMARY_DIR, OUT = args.source, args.output
    ZIKA_C = SUMMARY_DIR / "Figure4_zika_315_continuous_monthly.tsv"
    panels = load_panels()
    fig, _ = build_figure(panels)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Figure4"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches=None)
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, facecolor="white", bbox_inches=None)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches=None)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
