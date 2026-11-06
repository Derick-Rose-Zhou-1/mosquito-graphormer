import unittest

import torch

from mosquito_graphormer.baseline_temporal import TemporalTCNBaseline
from mosquito_graphormer.baseline_seir import GraphormerSEIRObservation


class BaselineTests(unittest.TestCase):
    def test_comparators_have_finite_outputs_and_gradients(self):
        torch.manual_seed(3)
        x = torch.randn(2, 14, 3, 13)
        cities = torch.arange(3)
        models = [TemporalTCNBaseline(13, 3), GraphormerSEIRObservation(13, 3)]
        for model in models:
            with self.subTest(model=type(model).__name__):
                if isinstance(model, TemporalTCNBaseline):
                    output = model(x, cities)
                else:
                    output = model(x, torch.ones(2, 14, 3), torch.full((2, 3), 1000.),
                                   cities, torch.ones(3, 3), torch.eye(3))
                self.assertEqual(tuple(output.shape), (2, 3, 7))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue((output >= 0).all())
                output.mean().backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))


if __name__ == "__main__":
    unittest.main()
