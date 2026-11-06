"""Artificial figure inputs only; no study observations."""
import unittest
import pandas as pd
from mosquito_graphormer.figures.figure3 import prepare

class FigureContracts(unittest.TestCase):

    def test_figure3_requires_matched_seed_targets(self):
        rows = [dict(seed=seed, city='广州市', origin=f'2025-07-{d}', horizon=h, target_date=str((pd.Timestamp(f'2025-07-{d}') + pd.Timedelta(days=h)).date()), observed=1.0, prediction=2.0) for seed in range(2025, 2035) for d in (18, 19, 20, 21) for h in range(1, 8)]
        history = pd.DataFrame(dict(city='广州市', date=pd.date_range('2025-07-01', periods=21), cases=1.0))
        frame = pd.DataFrame(rows)
        _, _, inputs, errors = prepare(frame, history)
        self.assertEqual(inputs, [7.0] * 4)
        self.assertEqual(errors, [1.0] * 4)
        frame.loc[0, 'observed'] = 3.0
        with self.assertRaises(ValueError):
            prepare(frame, history)
