# Matched components, comparators and spatial sensitivity

These modules contain code only. No surveillance records, predictions,
checkpoints or movement matrices are distributed.

## Matched component settings

`ablations.arm_config` implements FULL, NO_WEATHER_ENCODER, NO_MOBILITY,
NO_GIS, NO_TCN and NO_GRAPHORMER. The weather intervention removes only the
four direct weather features from the temporal encoder; climate-derived
mechanistic priors remain. The Graphormer intervention leaves mechanistic
spatial coupling in place. These are component interventions, not replacements
of the whole architecture. No combined weather-removal/FREE-parameter arm is
included.

`scripts/run_panel.py --arm ...` applies these settings to the existing generic
train/development/test workflow. This is **not** the study-specific epoch
selection, refitting and frozen-checkpoint protocol used for the matched R3
experiment. Supplying an arm alone does not reproduce that experiment.

For already generated matched predictions:

```bash
python scripts/summarize_ablations.py --input predictions.csv --output summaries --seeds 1 2 3 4 5 6 7 8 9 10
```

Input columns: `arm,seed,origin,target_date,horizon,city,observed,prediction`.
Supply the actual seeds, not necessarily those illustrative integers. All arms
and seeds must contain identical forecast slots and observations. Dates must
satisfy target_date = origin + horizon days. Metrics pool all supplied slots
equally within a seed. The program outputs means and sample standard deviations
of seed-specific metrics, paired ablation-minus-full differences, and **separate**
metrics of ensemble-mean predictions. The latter are not substitutes for seed
means. This statistical check does not prove that input checkpoints were trained
on the same panel or without test leakage; audit training provenance separately.

## Comparators

`baseline_temporal.TemporalTCNBaseline` and
`baseline_seir.GraphormerSEIRObservation` preserve the archived comparator
class implementations used by the final scenario runner. They do not import the
old runner's obsolete split constants or server paths.

Both take features `[batch,history,cities,features]` and return nonnegative
predictions `[batch,cities,horizon]`. City indices must match embedding order.
The SEIR comparator additionally takes untransformed count history, population
in **persons** (not log population), a `[cities,cities]` adjacency mask and a
row-normalized `[cities,cities]` movement matrix, with origin rows and destination
columns. Its observation head is part of training. Do not inverse-standardize
its output as though it were standardized features. The archived comparator
input has 13 features, whereas the main-model panel contract has 8; do not
silently interchange these input pipelines.

Only model classes are exposed here. Comparator-specific training, train-only
feature scaling and scenario inputs remain necessary for exact reproduction.

## Excluding a high-burden city

```bash
python scripts/summarize_spatial.py --input city_summary.csv --output sensitivity.csv --exclude-city CITY_LABEL
```

Input columns: `panel,city,observed,predicted_mean`, one row per panel and city.
The input must already use the intended forecast-horizon and forecast-origin
aggregation and ensemble averaging. This is a post hoc exclusion of city pairs,
not retraining with a city removed. Pearson and Spearman correlations are
calculated before and after exclusion; nominal P-values do not adjust for
spatial dependence or post hoc selection. Missing exclusion labels are errors.

## Scope still outside this release

The complete study-specific data preparation, R3 select/refit/freeze runner,
comparator training loops, simulation generation, R4 aggregation and exact
manuscript figure assembly have not all been packaged into a portable public
workflow. Existing local experiment scripts cannot simply be copied: they also
contain private paths, legacy settings and data dependencies. This repository
does not claim complete independent reproduction of every manuscript panel.
