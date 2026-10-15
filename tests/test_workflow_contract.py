import tempfile
import unittest
from pathlib import Path

import numpy as np

from mosquito_graphormer.workflow import load_panel
from mosquito_graphormer.preprocessing import row_normalise_movement, temperature_proxy


class WorkflowContractTests(unittest.TestCase):
    def test_zero_movement_row_and_origin_direction(self):
        matrix = np.array([[9.0, 2.0, 2.0], [0.0, 4.0, 0.0], [1.0, 0.0, 7.0]])
        relative = row_normalise_movement(matrix)
        np.testing.assert_allclose(relative[0], [0, 0.5, 0.5])
        np.testing.assert_allclose(relative[1], [0, 0, 0])
        np.testing.assert_allclose(relative[2], [1, 0, 0])

    def test_temperature_proxy_reference_point(self):
        prior = temperature_proxy(np.array([300.0]))
        self.assertAlmostEqual(float(prior["prior_a"][0]), 0.30)
        self.assertAlmostEqual(float(prior["prior_mu_v"][0]), 0.10)
        self.assertAlmostEqual(float(prior["proxy_M"][0]), 300.0)

    def test_rejects_missing_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.npz"
            np.savez_compressed(path, x=np.zeros((3, 14, 2, 8)))
            with self.assertRaisesRegex(ValueError, "Missing panel arrays"):
                load_panel(path)


if __name__ == "__main__":
    unittest.main()
