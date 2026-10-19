# Analysis and interval definitions

This is a code-to-method map. Study inputs, fitted weights, observations,
split manifests and figure source data are held outside the public repository.
The generic training example is not an exact replay of the article experiments.

| Analysis | Prediction unit | Model fits | Displayed interval |
| --- | --- | --- | --- |
| 1 | Subset of a jointly fitted city panel | Ten independently initialised model runs | Empirical 2.5th–97.5th percentiles across aligned predictions |
| 2 | Full jointly fitted city panel | Ten independently initialised model runs | Empirical 2.5th–97.5th percentiles across aligned predictions |
| 3 | Four fixed forecast origins from a separately fitted pre-forecast model | Ten independently initialised model runs | Empirical 2.5th–97.5th percentiles across aligned predictions |
| 4 | Municipality predictions from department-specific model fits, aggregated after equal-weight prediction averaging | Ten independently initialised model runs per task | Empirical 2.5th–97.5th percentiles across aligned predictions |
| 5 | One synthetic epidemic, five reporting-input conditions, common continuous simulated incidence target | Ten independently initialised model runs per condition and model | Two-sided 95% t intervals for the mean run-level metric |

In analyses 1–4 the prediction ranges describe between-model spread; they are
not calibrated predictive intervals. In analysis 5 the t intervals concern
the mean metric across model initialisations while the epidemic inputs and
reporting perturbation are held fixed. They do not represent ten independent
epidemics or uncertainty in transmission and reporting parameters.

The fifth analysis generated a continuous exposed-to-infectious transition
flow and a separate Poisson-sampled observation stream. The unperturbed
continuous flow served as the shared reference target. Each reporting
condition changed the available case-input history only; the Poisson stream
was not an evaluation target. Keeping the reference target fixed removes an
additional observation-sampling step from the scenario comparison. It also
limits the interpretation: this analysis does not estimate accuracy against
realised reported counts.

The public `summarize_scenarios.py` entry point validates target-key equality
and computes the fifth analysis's per-run H1–H7 MAE, RMSE, Pearson, R² and
top-10 capture from external prediction files. Exact article figures and
article-level numerical results require the controlled study inputs and
source files held by the data providers.

The generic `run_panel.py` defaults are for an execution example. The actual
first and second analysis runners used batch size 4, at most 60 epochs,
patience 12, learning rate 3e-4, weight decay 0.005 and gradient clip norm 1.
The fourth analysis used department-specific fits, batch size 16 and the
shared configuration's at most 100 epochs and patience 20; the same learning
rate, weight decay and clip norm applied. The third analysis involved a
separate pre-forecast fit and subsequent frozen forecasts. The fifth analysis
reused training origins for checkpoint monitoring, so it must not be run
through the generic loader's distinct-development-split contract.
