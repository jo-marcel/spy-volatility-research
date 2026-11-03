"""Petits tests compréhensibles, sans pytest."""
import unittest
import numpy as np
from data_loader import load_prices
from features import add_features, research_data
from split_data import split_train_test
from metrics import calculate_metrics


class ProjectTests(unittest.TestCase):
    def test_features_do_not_use_future(self):
        prices = load_prices()
        short = add_features(prices.iloc[:100])
        long = add_features(prices)
        # Le dernier jour de la version courte ne connaît pas sa cible.
        self.assertTrue(np.isnan(short["next_day_abs_return"].iloc[-1]))
        # La volatilité de cette date reste identique quand on ajoute le futur.
        self.assertAlmostEqual(short["volatility_20d"].iloc[-1],
                               long["volatility_20d"].iloc[99])

    def test_split_does_not_overlap(self):
        data = research_data(add_features(load_prices()))
        train, test = split_train_test(data)
        self.assertLess(train.index.max(), test.index.min())
        self.assertEqual(len(train), 1993)
        self.assertEqual(len(test), 751)

    def test_metrics(self):
        result = calculate_metrics([1, 2], [1, 4])
        self.assertAlmostEqual(result["mae"], 1)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
