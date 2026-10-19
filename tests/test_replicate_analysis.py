import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mosquito_graphormer.evaluation import top_k_capture
from mosquito_graphormer.replicate_analysis import summarise_scenarios, t_interval


class ReplicateAnalysisTests(unittest.TestCase):
    def test_t_interval_is_interval_for_mean(self):
        values = np.array([1.0, 2.0, 3.0])
        mean, low, high = t_interval(values)
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual((low + high) / 2, mean)
        self.assertGreater(high - low, 0)

    def test_top_k_matches_archived_argsort(self):
        observed = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        predicted = observed[::-1]
        expected = len(set(np.argsort(observed)[-10:]) &
                       set(np.argsort(predicted)[-10:])) / 10
        self.assertEqual(top_k_capture(observed, predicted, 10), expected)

    def test_shared_targets_are_checked_across_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (1, 2):
                path = root / f"seed_{seed}" / "sparse" / "outputs" / "model" / "predictions.csv"
                path.parent.mkdir(parents=True)
                pd.DataFrame({
                    "city": [f"site_{i:02d}" for i in range(12)],
                    "origin_date": ["origin"] * 12,
                    "target_date": ["target"] * 12,
                    "horizon": [7] * 12,
                    "observed": np.arange(12, dtype=float),
                    "predicted": np.arange(12, dtype=float) + seed,
                }).to_csv(path, index=False)
            audit = summarise_scenarios(root, root / "summary", seeds=(1, 2),
                                        scenarios=("sparse",), models=("model",))
            self.assertEqual(audit["shared_target_keys"], 12)
            self.assertEqual(audit["model_runs"], 2)
            table = pd.read_csv(root / "summary" / "MEAN_T_INTERVALS.csv")
            self.assertAlmostEqual(float(table.loc[0, "MAE_mean"]), 1.5)
            changed = root / "seed_2" / "sparse" / "outputs" / "model" / "predictions.csv"
            frame = pd.read_csv(changed)
            frame.loc[0, "observed"] = 99.0
            frame.to_csv(changed, index=False)
            with self.assertRaisesRegex(ValueError, "Test target keys or observed values differ"):
                summarise_scenarios(root, root / "summary", seeds=(1, 2),
                                    scenarios=("sparse",), models=("model",))


if __name__ == "__main__":
    unittest.main()
