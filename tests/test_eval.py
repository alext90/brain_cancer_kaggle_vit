import unittest

import numpy as np

from src.eval import calc_metrics


class TestEvaluateModel(unittest.TestCase):
    def test_calc_metrics(self):
        # Sample predictions and labels
        all_preds = [0, 1, 1, 0, 1]
        all_labels = [0, 1, 0, 0, 1]

        # Call the function
        f1, acc, cm = calc_metrics(all_preds, all_labels)

        # Check the output types
        self.assertIsInstance(f1, float)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(cm, np.ndarray)

        # Check the shape of confusion matrix
        self.assertEqual(cm.shape, (2, 2))
