"""Small artificial arrays only; no observed surveillance records."""
import unittest

import numpy as np
import pandas as pd

from mosquito_graphormer.ablations import ARMS, arm_config
from mosquito_graphormer.config import model_config
from mosquito_graphormer.ablation_analysis import summarize_ablations
from mosquito_graphormer.spatial_analysis import spatial_sensitivity


def fixture():
    return pd.DataFrame([
        dict(arm=arm, seed=seed, origin="2000-01-01", target_date="2000-01-02",
             horizon=1, city=str(city), observed=float(city + 2),
             prediction=float(city + 2 + offset))
        for arm in ARMS for seed, offset in [(1, -1), (2, 1)] for city in range(3)])


class MatchedTests(unittest.TestCase):
    def test_seed_metrics_are_not_ensemble_metrics(self):
        result = summarize_ablations(fixture(), [1, 2])
        np.testing.assert_array_equal(result["seed_mean_sd"].mae_mean, np.ones(6))
        np.testing.assert_array_equal(result["ensemble_metrics"].mae, np.zeros(6))
        np.testing.assert_array_equal(result["paired_differences"].mae_ablation_minus_full, np.zeros(10))

    def test_alignment_failures_are_rejected(self):
        for kind in ("missing", "duplicate", "target", "date"):
            frame = fixture()
            if kind == "missing": frame = frame.iloc[1:]
            if kind == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
            if kind == "target": frame.loc[0, "observed"] += 1
            if kind == "date": frame.loc[0, "target_date"] = "2000-01-03"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                summarize_ablations(frame, [1, 2])

    def test_weather_arm_retains_climate_priors(self):
        base = model_config()
        cfg = arm_config(base, "NO_WEATHER_ENCODER")
        self.assertEqual(len(base["temporal_feature_names"]), 8)
        self.assertEqual(len(cfg["temporal_feature_names"]), 4)
        self.assertEqual(cfg["parameter_mode"], base["parameter_mode"])
        with self.assertRaises(ValueError): arm_config(base, "FREE")

    def test_exclusion_and_invalid_city(self):
        frame = pd.DataFrame(dict(panel=["g"] * 4, city=list("abcd"),
                                  observed=[1., 2., 3., 100.], predicted_mean=[3., 2., 1., 100.]))
        result = spatial_sensitivity(frame, ["d"])
        self.assertAlmostEqual(result.iloc[1].pearson, -1)
        self.assertEqual(result.iloc[1].n_cities, 3)
        with self.assertRaises(ValueError): spatial_sensitivity(frame, ["missing"])


if __name__ == "__main__":
    unittest.main()
