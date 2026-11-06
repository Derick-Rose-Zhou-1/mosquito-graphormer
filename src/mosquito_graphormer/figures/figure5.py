"""Final b–d rendering. Signed Pearson values remain in the source output.

The submitted display clips negative correlations at zero. This preserves the
selected figure convention; it does not change the computed correlations.
"""
from pathlib import Path


def render(source_path, output_dir):

    import importlib.util
    import json
    from pathlib import Path
    import os

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd
    from scipy.stats import t as student_t


    INFILE = source_path
    OUT = output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    from .style import apply_style
    apply_style((7.0, 6.4, 6.0))
    mpl.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "savefig.facecolor": "white",
    })

    SCENARIOS = [
        ("sparse", "Sparse early observations"),
        ("missing20", "Missing reports"),
        ("delay3", "Reporting delay"),
        ("resource", "Resource-stratified\nunderreporting"),
        ("spatial", "Spatial surveillance\nheterogeneity"),
    ]
    MODELS = ["Full", "Graphormer-SEIR", "TCN"]
    DISPLAY = {
        "Full": "Mosquito-Graphormer",
        "Graphormer-SEIR": "Graphormer combined with SEIR",
        "TCN": "TCN",
    }
    COLORS = {"Full": "#C51B3A", "Graphormer-SEIR": "#55585A", "TCN": "#2E73B8"}
    OFFSETS = {"Full": 0.22, "Graphormer-SEIR": 0.0, "TCN": -0.22}
    BAR_H = 0.15
    N = 10

    df = pd.read_csv(INFILE)
    df = df[(df["horizon"] == 7) & (df["scenario"].isin([s[0] for s in SCENARIOS]))]
    assert len(df) == 15, f"Expected 5 scenarios x 3 models at H7; found {len(df)} rows"
    assert set(df["model"]) == set(MODELS)
    assert (df["n"] == N).all()
    assert df["observed_total"].nunique() == 1

    def cell(scenario: str, model: str) -> pd.Series:
        x = df[(df.scenario == scenario) & (df.model == model)]
        assert len(x) == 1
        return x.iloc[0]

    def ci_half(sd: float) -> float:
        return float(student_t.ppf(0.975, N - 1) * sd / np.sqrt(N))

    # Auditable source data: plotted means and their seed-level 95% t intervals.
    source_rows = []
    for order, (scenario, label) in enumerate(SCENARIOS, start=1):
        for model in MODELS:
            r = cell(scenario, model)
            for metric, panel in [("MAE", "b"), ("Pearson", "c"), ("Top10", "d")]:
                mean, sd = float(r[f"{metric}_mean"]), float(r[f"{metric}_sd"])
                half = ci_half(sd)
                source_rows.append({
                    "panel": panel,
                    "scenario_order": order,
                    "scenario_code": scenario,
                    "scenario_label": label.replace("\n", " "),
                    "model": DISPLAY[model],
                    "metric": metric,
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                    "n_model_seeds": N,
                    "forecast_horizon": "H7",
                    "test_origin": "2024-07-14",
                    "target_date": "2024-07-21",
                    "cities": 371,
                    "epidemic_replicates": 1,
                    "owning_section": "Result 5; 14-day six-condition, ten-seed comparison",
                })
    source = pd.DataFrame(source_rows)
    source_path = OUT / "FIGURE5_BCD_TRAIN14_SOURCE_DATA.csv"
    source.to_csv(source_path, index=False, float_format="%.10g")

    fig = plt.figure(figsize=(7.45, 3.65), facecolor="white")
    outer = fig.add_gridspec(
        1, 3, left=0.205, right=0.96, top=0.88, bottom=0.23,
        wspace=0.38,
    )
    ax_b = fig.add_subplot(outer[0, 0])
    ax_c = fig.add_subplot(outer[0, 1])
    ax_d = fig.add_subplot(outer[0, 2])
    axes = [ax_b, ax_c, ax_d]
    ybase = np.arange(len(SCENARIOS))[::-1]
    for ax in axes:
        ax.set_ylim(-0.55, len(SCENARIOS) - 0.45)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", direction="out", length=3, width=0.6)
        ax.spines["bottom"].set_linewidth(0.65)
        ax.spines["left"].set_linewidth(0.65)

    # Panel b. log1p bar lengths retain zero while compressing the large comparator errors.
    for model in MODELS:
        means = np.array([float(cell(s, model).MAE_mean) for s, _ in SCENARIOS])
        sds = np.array([float(cell(s, model).MAE_sd) for s, _ in SCENARIOS])
        yy = ybase + OFFSETS[model]
        for mean, sd, y in zip(means, sds, yy):
            half = ci_half(sd)
            lo, hi = mean - half, mean + half
            ax_b.barh(y, np.log1p(mean), height=BAR_H, color=COLORS[model], edgecolor="white", linewidth=0.25, zorder=2)
            low_t, high_t = np.log1p(max(0.0, lo)), np.log1p(hi)
            ax_b.hlines(y, low_t, high_t, color=COLORS[model], linewidth=0.85, zorder=3)
            ax_b.vlines([low_t, high_t], y - BAR_H * 0.29, y + BAR_H * 0.29,
                        color=COLORS[model], linewidth=0.8, zorder=3)
    mae_ticks = np.array([0, 1, 5, 20, 100, 500], dtype=float)
    ax_b.set_xlim(0, np.log1p(500))
    ax_b.set_xticks(np.log1p(mae_ticks))
    ax_b.set_xticklabels([str(int(v)) for v in mae_ticks])
    ax_b.set_yticks(ybase)
    ax_b.set_yticklabels([label for _, label in SCENARIOS])
    ax_b.tick_params(axis="y", labelsize=6.0)
    ax_b.text(-0.045, 1.075, "b", transform=ax_b.transAxes, fontsize=8.0, fontweight="bold",
               va="bottom", ha="left", color="#202124")
    ax_b.text(0.075, 1.075, "Forecast error", transform=ax_b.transAxes, fontsize=7.0,
               fontweight="bold", va="bottom", ha="left", color="#202124")

    # Panel c. Clip negative values at the zero boundary for display only.
    for model in MODELS:
        means = np.array([float(cell(s, model).Pearson_mean) for s, _ in SCENARIOS])
        sds = np.array([float(cell(s, model).Pearson_sd) for s, _ in SCENARIOS])
        yy = ybase + OFFSETS[model]
        for mean, sd, y in zip(means, sds, yy):
            half = ci_half(sd)
            lo, hi = mean - half, mean + half
            if mean >= 0:
                ax_c.barh(y, mean, height=BAR_H, color=COLORS[model], edgecolor="white", linewidth=0.25, zorder=2)
                clipped_lo, clipped_hi = max(0.0, lo), min(1.0, hi)
                ax_c.hlines(y, clipped_lo, clipped_hi, color=COLORS[model], linewidth=0.85, zorder=3)
                ax_c.vlines([clipped_lo, clipped_hi], y - BAR_H * 0.29, y + BAR_H * 0.29,
                            color=COLORS[model], linewidth=0.8, zorder=3)
            else:
                # Zero-height visual bar; the source table retains the measured negative r.
                ax_c.plot([0], [y], marker="|", markersize=BAR_H * 16,
                          color=COLORS[model], zorder=4)
    ax_c.set_xlim(0, 1.0)
    ax_c.set_xticks(np.linspace(0, 1, 6))
    ax_c.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax_c.set_yticks(ybase)
    ax_c.set_yticklabels([])
    ax_c.tick_params(axis="y", labelleft=False, left=False)
    ax_c.text(-0.045, 1.075, "c", transform=ax_c.transAxes, fontsize=8.0, fontweight="bold",
              va="bottom", ha="left", color="#202124")
    ax_c.text(0.075, 1.075, "Pearson", transform=ax_c.transAxes, fontsize=7.0,
              fontweight="bold", va="bottom", ha="left", color="#202124")
    ax_c.set_xlabel("Pearson correlation", labelpad=3)

    # Panel d. Top-10 capture is displayed on the full proportion scale.
    for model in MODELS:
        means = np.array([float(cell(s, model).Top10_mean) for s, _ in SCENARIOS])
        sds = np.array([float(cell(s, model).Top10_sd) for s, _ in SCENARIOS])
        yy = ybase + OFFSETS[model]
        for mean, sd, y in zip(means, sds, yy):
            half = ci_half(sd)
            ax_d.barh(y, mean, height=BAR_H, color=COLORS[model], edgecolor="white", linewidth=0.25, zorder=2)
            lo, hi = max(0.0, mean - half), min(1.0, mean + half)
            ax_d.hlines(y, lo, hi, color=COLORS[model], linewidth=0.85, zorder=3)
            ax_d.vlines([lo, hi], y - BAR_H * 0.29, y + BAR_H * 0.29, color=COLORS[model], linewidth=0.8, zorder=3)
    ax_d.set_xlim(0, 1.0)
    ax_d.set_xticks(np.linspace(0, 1, 6))
    ax_d.set_xticklabels([f"{int(x * 100)}%" for x in np.linspace(0, 1, 6)])
    ax_d.set_yticks(ybase)
    ax_d.set_yticklabels([])
    ax_d.tick_params(axis="y", labelleft=False, left=False)
    ax_d.text(-0.045, 1.075, "d", transform=ax_d.transAxes, fontsize=8.0, fontweight="bold",
              va="bottom", ha="left", color="#202124")
    ax_d.text(0.075, 1.075, "High-risk city detection", transform=ax_d.transAxes, fontsize=7.0,
              fontweight="bold", va="bottom", ha="left", color="#202124")
    ax_d.set_xlabel("Top 10 capture rate", labelpad=3)

    ax_b.set_xlabel("Mean absolute error (cases; log1p scale)", labelpad=3)

    handles = [Line2D([0], [0], color=COLORS[m], linewidth=5, solid_capstyle="butt", label=DISPLAY[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.64, 0.035), ncol=3,
               fontsize=6.4, handletextpad=0.5, columnspacing=1.1, frameon=False)

    stem = OUT / "Figure5_bcd"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04)

    # Check visible text geometry before export and record provenance for this preview.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_artists = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    visible_text = [t.get_text() for t in text_artists]
    boxes = [(t, t.get_window_extent(renderer)) for t in text_artists]
    overlaps = []
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if ba.overlaps(bb):
                overlaps.append((a.get_text(), b.get_text()))
    assert not overlaps, f"Visible text overlap detected: {overlaps}"
    outside = [(t.get_text(), t.get_window_extent(renderer).bounds) for t in text_artists
               if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0])
               or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
    assert not outside, f"Visible text outside canvas: {outside}"
    qa = {
        "status": "rendered_preview",
        "source_table": str(INFILE),
        "source_rows": int(len(df)),
        "panel_data": "H7, 14-day training protocol, five scenarios, three models",
        "test_origin": "2024-07-14",
        "target_date": "2024-07-21",
        "cities": 371,
        "model_seeds": list(range(3101, 3111)),
        "epidemic_replicates": 1,
        "interval": "two-sided 95% Student-t CI across ten model seeds",
        "mae_axis": "log1p-transformed bar lengths; axis begins at 0 and tick labels show original MAE in cases",
        "pearson_axis": "0 to 1.0; negative means displayed at zero only, with original signed values retained in source data",
        "files": [p.name for p in [stem.with_suffix(".pdf"), stem.with_suffix(".svg"), stem.with_suffix(".png"), stem.with_suffix(".tiff"), source_path]],
        "scenario_alignment_note": "Scenario names and five-condition row order match the previous Figure 5B-D artwork; the one-day reporting-delay condition is omitted.",
        "geometry_check": {"visible_text_overlap_count": len(overlaps), "text_outside_canvas_count": len(outside)},
        "text_labels": visible_text,
    }
    (OUT / "FIGURE5_BCD_TRAIN14_QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.close(fig)
    print(json.dumps(qa, ensure_ascii=True))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)


if __name__ == "__main__":
    main()
