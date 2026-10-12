import unittest

import torch

from mosquito_graphormer import CoVFourLoss, FEATURE_NAMES, ForecastBackbone, MosquitoGraphormer, model_config


class ModelTests(unittest.TestCase):
    def test_forecast_and_mechanism_gradient(self):
        torch.manual_seed(11)
        names = list(FEATURE_NAMES)
        model = MosquitoGraphormer(ForecastBackbone(model_config(), len(names), 2, names))
        batch = {
            "x": torch.zeros(1, 14, 2, len(names)),
            "history": torch.ones(1, 14, 2),
            "mobility": torch.ones(1, 14, 2, 2),
            "distance": torch.ones(2, 2),
            "geo_support": torch.ones(2, 2, dtype=torch.bool),
            "population": torch.log1p(torch.full((1, 2), 1000.0)),
            "prior_a": torch.full((1, 2), 0.3),
            "prior_sigma_v": torch.full((1, 2), 0.2),
            "prior_mu_v": torch.full((1, 2), 0.1),
            "prior_M_z": torch.zeros(1, 2),
        }
        forecast = model(batch)
        self.assertEqual(tuple(forecast["raw_prediction"].shape), (1, 2, 7))
        self.assertTrue(forecast["numerical_qc"]["pass"])
        objective = CoVFourLoss()
        total, detail = objective(forecast, torch.ones(1, 2, 7))
        self.assertEqual(list(detail["weights"].values()), [0.25] * 4)
        total.backward()
        self.assertGreater(float(model.backbone.parameter_head.delta.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
