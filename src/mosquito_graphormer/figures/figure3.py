"""Final a–c Guangzhou layout, calculated from external predictions and history."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

from ..ablation_analysis import KEYS


def prepare(predictions, history):
    frame = predictions.copy()
    if "arm" in frame:
        frame = frame[frame.arm == "FULL"].copy()
    required = {"seed", *KEYS, "observed", "prediction"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Missing prediction columns or values")
    if frame.duplicated(["seed", *KEYS]).any():
        raise ValueError("Duplicate forecast slots")
    if set(frame.seed) != set(range(2025, 2035)) or frame.city.nunique() != 1:
        raise ValueError("Expected one city and all ten selected seeds")
    if frame.city.iloc[0] != "广州市":
        raise ValueError("Figure 3 is the Guangzhou experiment")
    means = frame.groupby(KEYS).agg(observed=("observed", "first"),
                                     prediction=("prediction", "mean"),
                                     n=("seed", "nunique"), n_observed=("observed", "nunique")).reset_index()
    if len(means) != 28 or not means.n.eq(10).all() or not means.n_observed.eq(1).all():
        raise ValueError("Unmatched ten-seed observations or forecast slots")
    origins = [f"2025-07-{day}" for day in (18, 19, 20, 21)]
    if sorted(means.origin.unique()) != origins:
        raise ValueError("Unexpected forecast origins")
    history = history[history.city == "广州市"].copy()
    history["date"] = pd.to_datetime(history.date)
    if history.date.duplicated().any():
        raise ValueError("Duplicate history dates")
    histories, errors = [], []
    for origin in origins:
        g = means[means.origin == origin].sort_values("horizon")
        targets = pd.to_datetime(g.target_date)
        if list(g.horizon) != list(range(1, 8)) or not np.array_equal(
                (targets - pd.Timestamp(origin)).dt.days.to_numpy(), np.arange(1, 8)):
            raise ValueError("Forecast horizon/date mismatch")
        # Panel a displays seven reported days; the training model still uses 14.
        dates = pd.date_range(end=origin, periods=7)
        h = history.set_index("date").reindex(dates).cases
        if h.isna().any() or not np.isfinite(h).all() or (h < 0).any():
            raise ValueError("Incomplete displayed history; no implicit zero filling")
        histories.append(float(h.sum()))
        errors.append(float(np.abs(g.prediction - g.observed).mean()))
    values = means[["observed", "prediction"]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not np.isfinite(histories).all():
        raise ValueError("Invalid counts")
    return means, origins, histories, errors


def render(predictions, history, output):
    means, origins, inputs, errors = prepare(predictions, history)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(862.56 / 72, 266 / 72), facecolor="white")
    left = fig.add_axes([.086, .236, .124, .646])
    centre = fig.add_axes([.304, .236, .487, .646])
    right = fig.add_axes([.848, .236, .145, .646])
    palette = ["#005596", "#CA6E28", "#7BA6D8", "#C0504D"]
    y = np.arange(3, -1, -1)
    for ax, values in [(left, inputs), (right, errors)]:
        ax.barh(y, values, height=.42, color=palette)
        for pos, value in zip(y, values):
            label = f"{value:g}" if ax is left else f"{value:.2f}"
            ax.text(value + .65, pos, label, va="center", color="#304653")
        ax.set_ylim(-.5, 3.5)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    left.set_xlim(0, 26); right.set_xlim(0, 33)
    left.set_yticks(y, [f"{pd.Timestamp(o).day} July" for o in origins])
    right.set_yticks([])
    left.set_xlabel("Cases", fontsize=10)
    right.set_xlabel("Cases/day", fontsize=10)
    blue = LinearSegmentedColormap.from_list("prediction", ["#edf3fb", "#8fb3df", "#005596"])
    orange = LinearSegmentedColormap.from_list("observation", ["#fff7f0", "#e7b28c", "#ca6e28"])
    norm = Normalize(0, 45)
    for pos, origin in zip(y, origins):
        g = means[means.origin == origin].sort_values("horizon")
        for offset, field, cmap, label in [(.21, "prediction", blue, "Pred."),
                                            (-.21, "observed", orange, "Obs.")]:
            centre.text(18.35, pos + offset, label, ha="right", va="center",
                        color="#005596" if field == "prediction" else "#ca6e28", fontsize=8.5)
            for row in g.itertuples():
                day = pd.Timestamp(row.target_date).day
                value = getattr(row, field)
                centre.add_patch(Rectangle((day - .5, pos + offset - .195), 1, .39,
                                          facecolor=cmap(norm(value)), edgecolor="white", linewidth=.8))
                centre.text(day, pos + offset, f"{value:.1f}" if field == "prediction" else f"{value:g}",
                            ha="center", va="center", fontsize=8.2,
                            color="white" if field == "prediction" and value >= 30 else "#304b60")
    centre.set(xlim=(18.5, 28.5), ylim=(-.5, 3.5), yticks=[], xticks=range(19, 29))
    centre.spines[:].set_visible(False); centre.tick_params(length=0)
    centre.set_xlabel("Target date (July 2025)", fontsize=10)
    for x, title in [(.086, "Reported input"), (.304, "Seven-day forecasts"), (.848, "MAE")]:
        fig.text(x, .935, title, fontsize=10)
    for x, letter in [(.008, "a"), (.247, "b"), (.812, "c")]:
        fig.text(x, .935, letter, fontsize=14, weight="bold")
    for x, cmap, label, color in [(.304, blue, "Predicted", "#005596"),
                                 (.584, orange, "Observed", "#ca6e28")]:
        cax = fig.add_axes([x, .068, .197, .025])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_ticks([0, 15, 30, 45]); cb.ax.tick_params(labelsize=7, length=2)
        cb.outline.set_visible(False)
        fig.text(x - .012, .08, label, ha="right", va="center", color=color, fontsize=8)
    fig.text(.802, .08, "Daily cases", va="center", color="#607080", fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output / f"Figure3.{ext}", dpi=300, bbox_inches=None)
    means.to_csv(output / "Figure3_source.csv", index=False)
    pd.DataFrame(dict(origin=origins, reported_input=inputs, mae=errors)).to_csv(
        output / "Figure3_window_summary.csv", index=False)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(pd.read_csv(args.predictions), pd.read_csv(args.history), args.output)


if __name__ == "__main__":
    main()
