# External-panel training workflow

This repository now includes a runnable data contract, CoV4 training loop,
development-checkpoint selection and H1–H7 evaluation. The runnable example
contains only generated values and no study records, locations or dates.
The reusable preprocessing functions are in
`src/mosquito_graphormer/preprocessing.py`.

```shell
python scripts/make_example_panel.py --output /path/to/example.npz
python scripts/run_panel.py --panel /path/to/example.npz --output /path/to/results --seed 1 --epochs 2
python scripts/plot_predictions.py --predictions /path/to/results/test_predictions.npz --output /path/to/results/curve.png
```

All generated files must be written to a caller-selected output directory.
The `.npz` panel has one row per forecast origin, with arrays documented in
`load_panel` in `src/mosquito_graphormer/workflow.py`. `split` uses 0 for
training, 1 for development and 2 for held-out test. The 14 input time steps
run oldest to newest. Targets are seven subsequent horizons. Movement arrays
have origin rows and destination columns. The supplied development split alone
selects the checkpoint. The test split is evaluated once after selection.
An optional Boolean `mask` array, shaped like the target, excludes unavailable
target entries from loss and metrics.

The example demonstrates the public training and evaluation functions. It
does not reconstruct the restricted study panels, fitted priors, protocol
manifests or article figures. Those materials are controlled by their data
holders. Manuscript estimates should not be attributed to the generated
example. The reported article analyses also use setting-specific model fits,
external inputs, split rules and aggregation steps not encoded in this generic
panel runner.

## Summarising the fifth analysis from externally supplied predictions

The fifth analysis holds one synthetic epidemic and its unperturbed continuous
simulated incidence target fixed. Five reporting-input scenarios are evaluated
with ten independently initialised model runs per model. The archived fifth
analysis reused training origins for checkpoint monitoring and had no separate
development window. This differs from the generic panel example above, which
requires a distinct development split.

The prediction-summary entry point expects externally supplied files at
`seed_<seed>/<scenario>/outputs/<model>/predictions.csv`. Each CSV contains
`city,origin_date,target_date,horizon,observed,predicted`. The `city` value is
only a matching key; the repository includes no real site identifiers or rows.
All runs must have identical target keys and observed values. Pass each seed,
scenario and model name as a repeated option:

```shell
python scripts/summarize_scenarios.py --runs /path/to/external/runs \
  --output /path/to/summary --seed 3101 --seed 3102 \
  --scenario sparse --model Full
```

For the manuscript comparison, supply all ten model seeds and all five
reporting conditions and three models. The output includes H1–H7 per-run
metrics, means, sample standard deviations, two-sided 95% t intervals for
each mean metric, a target-key audit and SHA-256 hashes of the prediction
inputs. The reported t intervals describe variability across model
initialisations within that fixed epidemic. They are neither prediction
intervals nor intervals across independently simulated epidemics. The top-10
calculation follows the archived NumPy `argsort` rule after sorting by site
key, including its tie behaviour.

`apply_reporting_scenario` in `src/mosquito_graphormer/surveillance.py`
implements the five reporting-input conditions using an externally supplied
synthetic daily panel. They are: unperturbed early history, independent
removal of 20% of eligible positive reports, a three-day report shift,
population-tier reporting at 0.90/0.70, and core/peripheral reporting at
1.00/0.30. Core membership and any focal site are caller inputs. The function
does not generate or change evaluation targets. Its outputs were checked on
synthetic data against the archived perturbation implementation.
