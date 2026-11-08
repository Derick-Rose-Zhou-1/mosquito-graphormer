# Mosquito-Graphormer

This repository contains the model implementations and code for data processing, the Colombia analyses, simulation experiments and figure generation.

## Contents

- `src/mosquito_graphormer/model.py`: shared backbone and Trend forecast readout.
- `src/mosquito_graphormer/mechanism.py`: coupled SEI–SEIR state updates and `SEISEIRHead`.
- `src/mosquito_graphormer/loss.py`: four-component coefficient-of-variation loss.
- `src/mosquito_graphormer/evaluation.py` and `plotting.py`: generic metrics and curve plotting.
- `src/mosquito_graphormer/replicate_analysis.py`: matched-target checks, per-run H1–H7 metrics and 95% t intervals for the mean metric across model initialisations.
- `src/mosquito_graphormer/preprocessing.py` and `workflow.py`: reusable input transforms, aligned-panel validation, training, development checkpoint selection and test evaluation.
- `src/mosquito_graphormer/surveillance.py`: reporting perturbations applied to an externally supplied synthetic epidemic while retaining its reference targets.
- `ablations.py`, `ablation_analysis.py` and `spatial_analysis.py`: matched-component settings, paired-seed summaries and city-exclusion analysis; see [ADDITIONAL_ANALYSES.md](ADDITIONAL_ANALYSES.md).
- `baseline_temporal.py` and `baseline_seir.py`: archived TCN and Graphormer–SEIR comparator architectures, with distinct input contracts documented in [ADDITIONAL_ANALYSES.md](ADDITIONAL_ANALYSES.md).
- `scripts/`: a runnable generated-panel example, training entry point, curve plotting entry point and prediction-summary entry point; see [WORKFLOW.md](WORKFLOW.md).
- `src/mosquito_graphormer/figures/`: manuscript figure rendering and PDF assembly.
- `tests/`: synthetic-input execution and gradient check.

Install the packages in `requirements.txt`, add `src` to `PYTHONPATH`, then run `python -m unittest discover -s tests -v`.

## License

The code in this repository is released under the [MIT License](LICENSE). Use, modification and redistribution, including commercial use, are permitted subject to the license terms.

## Data availability

The CDC-provided surveillance data used in this study cannot be publicly shared and are not included in this repository. The code license does not grant rights to these data.
