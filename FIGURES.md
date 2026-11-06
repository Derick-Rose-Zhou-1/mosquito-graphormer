# Manuscript plotting code

This release adds **rendering only**, using externally supplied final figure
source tables. No training code, observations, predictions, checkpoints, datasets,
manuscripts or generated figures are included. Keep all inputs and outputs outside
Git. Install requirements.txt and set PYTHONPATH to the absolute src directory.

## Figure 1 and Figure 5a

Export the external one-slide author artwork with installed Microsoft PowerPoint:

```powershell
./scripts/export_artwork.ps1 -InputPptx INPUT.pptx -OutputPdf OUTPUT.pdf
```

The artwork remains external; the script opens it read-only and exports a copy.

## Figure 2

Supply the selected final 612-row TSV. Install R packages ggplot2, dplyr, readr,
tidyr, stringr, forcats, scales, cowplot and ggrepel. Set FIGURE2_SOURCE to the TSV
and FIGURE2_OUT to an output directory. Run:

```text
Rscript scripts/plot_figure2.R
python -m mosquito_graphormer.figures.pdf_assembly --statistical PLOT.pdf --output FIGURE2.pdf
```

PLOT.pdf is Figure2_V4_CoV_trend_attachment_style.pdf. The second command sizes the
PDF to the submission page. Required source columns include panel, city,
date_or_window, model, metric, observed, predicted_mean, pred_lower, pred_upper.
The renderer retains supplied model rows and never substitutes a training archive.

## Figure 3

```text
python -m mosquito_graphormer.figures.figure3 --predictions SCORED.csv --history HISTORY.csv --output OUT
```

SCORED columns: seed, city, origin, target_date, horizon, observed, prediction;
optional arm selects FULL. Supply the ten matched seeds 2025–2034 for Guangzhou
origins July 18–21, 2025. HISTORY columns: city, date, cases. The final a–c layout
is reconstructed from the submitted artwork. Values are calculated from all ten
seeds; MAE follows ensemble averaging. Panel a displays seven reported days,
without altering the model's 14-day history. Typography is not pixel-identical
to the manually edited artwork.

## Figure 4

```text
python -m mosquito_graphormer.figures.figure4 --source SOURCE --output OUT
```

SOURCE must contain the existing final ensemble-mean tables:

- Figure4_bounded_adapter_national_summary.tsv
- Figure4_bounded_adapter_department_month_summary.tsv
- Figure4_bounded_adapter_peak_summary.tsv
- Figure4_bounded_adapter_cumulative_summary.tsv
- Figure4_zika_315_continuous_monthly.tsv

The Zika table includes plot_date and 17 months/489 target days, with 365 displayed
2016 target days. Numeric panels are preserved; later submission PDF text overlays
are not reproduced byte for byte.

## Figure 5

```text
python -m mosquito_graphormer.figures.figure5 --source H1_H5_H7_MEAN_SD.csv --output OUT
python -m mosquito_graphormer.figures.pdf_assembly --statistical OUT/Figure5_bcd.pdf --artwork PANEL_A.pdf --output FIGURE5.pdf
```

Source columns: scenario, model, horizon, n, observed_total and paired MAE_mean /
MAE_sd, Pearson_mean / Pearson_sd, Top10_mean / Top10_sd. Selected H7 rows cover
five scenarios and three models with ten seeds each. Intervals are 95% t intervals
for the mean run-level metric. Signed Pearson values remain in exported data;
the selected display clips negative bars at zero. Assembly requires external
Figure 5a artwork and does not recreate obsolete nested-PDF clipping artifacts.

## Verification and limits

Renderers were run locally with selected private inputs. Figure 3 window summaries
and Figure 4 numeric annotations agree with the final figures. Figure 5 b–d matched
the selected statistical-panel PDF pixel for pixel locally. PowerPoint export and
PDF assembly ran. Artificial tests check Figure 3 seed and target alignment.
No private verification inputs or outputs are published. Font/library changes may
alter layout. No manuscript figure or scientific result was replaced.
