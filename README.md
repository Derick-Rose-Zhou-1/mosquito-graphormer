# Mosquito-Graphormer

Compact source code for the selected temporal–graph forecasting architecture, its coupled SEI–SEIR head, four-component CoV loss, evaluation metrics and generic plotting.

## Contents

- `src/mosquito_graphormer/model.py`: shared backbone and Trend forecast readout.
- `src/mosquito_graphormer/mechanism.py`: coupled SEI–SEIR state updates and `SEISEIRHead`.
- `src/mosquito_graphormer/loss.py`: four-component coefficient-of-variation loss.
- `src/mosquito_graphormer/evaluation.py` and `plotting.py`: generic metrics and curve plotting.
- `tests/`: synthetic-input execution and gradient check.

Install the packages in `requirements.txt`, add `src` to `PYTHONPATH`, then run `python -m unittest discover -s tests -v`.

Inputs are supplied by the data holder. The model expects a 14-step history, aligned feature and movement tensors, a geographic support matrix, population on the `log1p(persons)` scale, and externally calculated mechanistic priors. Its output has seven forecast horizons. The geographic and movement inputs must use consistent node order and the same direction convention as the source data. No study observations, movement matrices, locations, dates, fitted weights, splits or figures are included.

The mobility window is ordered oldest to newest. Its non-negative exponential lag weights assign the greatest weight to the newest matrix. A synthetic test checks this direction.

This repository is a compact, name-cleaned implementation of the selected architecture. The changes were checked for output equivalence on synthetic input. It is **not a complete independent reproduction of the manuscript results**: those estimates require restricted inputs, study-specific preprocessing and training/evaluation protocols that are not published here. The plotting utility produces a generic aligned case curve, not the manuscript figures.
