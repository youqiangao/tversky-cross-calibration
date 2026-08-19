import sys
import types
import unittest
from unittest import mock

import torch

from tversky_cross_calibration import predict_rank


class RankPredictionTests(unittest.TestCase):
    def test_exact_dice_and_iou(self):
        probabilities = torch.tensor([[[0.8, 0.4, 0.1]]])
        self.assertEqual(predict_rank(probabilities, "dice").dtype, torch.bool)
        self.assertEqual(predict_rank(probabilities, "iou").shape, probabilities.shape)
        self.assertTrue(predict_rank(probabilities, "dice")[0, 0, 0])

    def test_valid_mask_is_excluded(self):
        probabilities = torch.tensor([[[0.9, 0.8, 0.7]]])
        valid = torch.tensor([[True, False, True]])
        prediction = predict_rank(probabilities, "dice", valid_mask=valid)
        self.assertFalse(prediction[0, 0, 1])

    def test_micro_pools_classes(self):
        probabilities = torch.tensor([[[0.9, 0.1], [0.8, 0.2]]])
        prediction = predict_rank(probabilities, "iou", aggregation="micro")
        self.assertEqual(prediction.shape, probabilities.shape)

    def test_large_dice_uses_official_ba(self):
        calls = []
        functional = types.ModuleType("rankseg.functional")

        def fake_rankseg(values, **kwargs):
            calls.append(kwargs)
            return values > 0.5

        functional.rankseg = fake_rankseg
        package = types.ModuleType("rankseg")
        package.functional = functional
        with mock.patch.dict(sys.modules, {"rankseg": package, "rankseg.functional": functional}):
            predict_rank(torch.tensor([[[0.9, 0.1]]]), "dice", exact_threshold=1)
        self.assertEqual(calls[0]["solver"], "BA")
        self.assertEqual(calls[0]["eps"], 1e-4)
        self.assertEqual(calls[0]["pruning_prob"], 0.0)

    def test_large_iou_uses_official_rma(self):
        calls = []
        functional = types.ModuleType("rankseg.functional")
        functional.rankseg = lambda values, **kwargs: calls.append(kwargs) or (values > 0.5)
        package = types.ModuleType("rankseg")
        package.functional = functional
        with mock.patch.dict(sys.modules, {"rankseg": package, "rankseg.functional": functional}):
            predict_rank(torch.tensor([[[0.9, 0.1]]]), "iou", exact_threshold=1)
        self.assertEqual(calls[0]["solver"], "RMA")
        self.assertNotIn("eps", calls[0])


if __name__ == "__main__":
    unittest.main()
