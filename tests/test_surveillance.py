import unittest

import numpy as np
import pandas as pd

from mosquito_graphormer.surveillance import apply_reporting_scenario


class ReportingScenarioTests(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame({
            "site": ["a"] * 5 + ["b"] * 5,
            "date": list(pd.date_range(pd.Timestamp(0), periods=5)) * 2,
            "cases": [1.4, 2.5, 3.1, 4.0, 5.0] * 2,
            "population": [100] * 5 + [200] * 5,
        })

    def test_delay_discards_unavailable_tail(self):
        result = apply_reporting_scenario(self.panel, "reporting_delay",
                                          known_start=pd.Timestamp(0), known_days=5, seed=1)
        np.testing.assert_allclose(result.loc[result.site == "a", "cases"], [0, 0, 0, 1.4, 2.5])

    def test_all_scenarios_preserve_reference_target(self):
        reference = self.panel["cases"].to_numpy().copy()
        for scenario in ("sparse_early", "missing_reports", "resource_underreporting",
                         "spatial_heterogeneity"):
            result = apply_reporting_scenario(self.panel, scenario,
                                              known_start=pd.Timestamp(0), known_days=5,
                                              seed=1, core_sites=["a"], focus_site="a")
            self.assertEqual(len(result), len(self.panel))
            self.assertTrue(np.array_equal(self.panel["cases"].to_numpy(), reference))


if __name__ == "__main__":
    unittest.main()
