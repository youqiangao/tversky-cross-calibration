from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from application.multilabel_segmentation.conventions import MACRO_CLASS_CONVENTION
from application.multilabel_segmentation.delta import _rows_match_cache, present_class_macro_delta
from application.multilabel_segmentation.metrics import RunningMultilabelMetrics
from application.multilabel_segmentation.rank_eval import (
    _rank_metrics_path,
    _rows_match_protocol,
    build_parser as build_multilabel_rank_parser,
)
from metrics import dice_coeff, iou
from scripts.build_paper_tables import build_table
from tversky_cross_calibration.config import dataset_config, paper_config
from tversky_cross_calibration.reproducibility import metadata_path, seed_everything, write_cache_metadata


ROOT = Path(__file__).resolve().parents[1]


def load_simulation(filename: str):
    path = ROOT / "simulation" / filename
    spec = importlib.util.spec_from_file_location(filename.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReproducibilityTests(unittest.TestCase):
    def test_seed_42_repeats_torch_and_numpy(self):
        seed_everything(42)
        first = (torch.rand(4), np.random.rand(4))
        seed_everything(42)
        second = (torch.rand(4), np.random.rand(4))
        self.assertTrue(torch.equal(first[0], second[0]))
        np.testing.assert_array_equal(first[1], second[1])

    def test_all_simulation_entrypoints_seed_42(self):
        for path in sorted((ROOT / "simulation").glob("sim-*.py")):
            source = path.read_text(encoding="utf-8")
            if 'if __name__ == "__main__":' in source:
                self.assertIn("seed_everything(42)", source, path.name)

    def test_cache_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "result.csv"
            csv_path.write_text("value\n1\n", encoding="utf-8")
            write_cache_metadata(csv_path, {"seed": 42})
            self.assertEqual(metadata_path(csv_path).read_text(encoding="utf-8"), '{\n  "seed": 42\n}\n')

    def test_paper_config_drives_dataset_roots(self):
        self.assertEqual(paper_config()["seed"], 42)
        self.assertEqual(dataset_config("cityscapes")["data_root"], "datasets/Cityscapes")


class MetricConventionTests(unittest.TestCase):
    def test_empty_denominator_is_zero(self):
        empty = torch.zeros((2, 1, 3, 3))
        self.assertTrue(torch.equal(dice_coeff(empty, empty), torch.zeros(2)))
        self.assertTrue(torch.equal(iou(empty, empty), torch.zeros(2)))

    def test_multilabel_macro_uses_present_ground_truth_classes(self):
        logits = torch.full((1, 3, 2, 2), -20.0)
        logits[:, 0] = 20.0
        targets = torch.zeros((1, 2, 2), dtype=torch.int64)
        valid = torch.ones_like(targets, dtype=torch.bool)
        metrics = RunningMultilabelMetrics(3)
        metrics.update(logits, targets, valid)
        result = metrics.compute_rounded(10)
        self.assertEqual(result["macro_dice"], 1.0)
        self.assertEqual(result["macro_iou"], 1.0)
        self.assertEqual(result["micro_dice"], 1.0)
        self.assertEqual(result["micro_iou"], 1.0)

    def test_absent_class_predictions_do_not_enter_macro_average(self):
        logits = torch.full((1, 3, 2, 2), -20.0)
        logits[:, 0] = 20.0
        logits[:, 2] = 20.0
        targets = torch.zeros((1, 2, 2), dtype=torch.int64)
        valid = torch.ones_like(targets, dtype=torch.bool)
        metrics = RunningMultilabelMetrics(3)
        metrics.update(logits, targets, valid)
        result = metrics.compute_rounded(10)
        self.assertEqual(result["macro_dice"], 1.0)
        self.assertEqual(result["macro_iou"], 1.0)
        self.assertAlmostEqual(result["micro_dice"], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(result["micro_iou"], 0.5, places=9)

    def test_multilabel_metrics_reject_all_ignore_image(self):
        logits = torch.zeros((1, 3, 2, 2))
        targets = torch.full((1, 2, 2), 255, dtype=torch.int64)
        valid = torch.zeros_like(targets, dtype=torch.bool)
        metrics = RunningMultilabelMetrics(3)
        with self.assertRaisesRegex(ValueError, "no valid pixels"):
            metrics.update(logits, targets, valid)

    def test_present_class_macro_delta_and_cache_convention(self):
        class_sums = torch.tensor([100.0, 50.0, 10.0])
        expected = (1.0 / 100.0 + 1.0 / 12.0) / 2.0
        self.assertAlmostEqual(present_class_macro_delta(class_sums, [0, 2], 12.0), expected)
        with self.assertRaisesRegex(ValueError, "no present ground-truth class"):
            present_class_macro_delta(class_sums, [], 12.0)
        current = [{"eta": "12.0", "macro_class_convention": MACRO_CLASS_CONVENTION}]
        legacy = [{"eta": "12.0"}]
        self.assertTrue(_rows_match_cache(current, 12.0))
        self.assertFalse(_rows_match_cache(legacy, 12.0))


class RankEvaluationProtocolTests(unittest.TestCase):
    def test_original_resolution_is_the_only_protocol(self):
        parser = build_multilabel_rank_parser()
        self.assertFalse(hasattr(parser.parse_args([]), "full_resolution"))

    def test_rank_cache_path_records_protocol_and_convention(self):
        full = _rank_metrics_path("out", "voc2012", "unet", "test", None)
        self.assertIn("original_present_ground_truth_macro", full.name)

    def test_rank_cache_requires_current_protocol(self):
        current = [{
            "evaluation_resolution": "original",
            "macro_class_convention": MACRO_CLASS_CONVENTION,
        }]
        self.assertTrue(_rows_match_protocol(current, "original"))
        self.assertFalse(_rows_match_protocol([{"evaluation_resolution": "original"}], "original"))


class BoundaryDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim4 = load_simulation("sim-4.py")
        cls.sim5 = load_simulation("sim-5.py")

    def test_sim4_reports_but_does_not_hide_violations(self):
        result = self.sim4.bound_diagnostics([0.1, 0.1], [0.2, 0.5], 0.0)
        self.assertEqual(result["num_points"], 2)
        self.assertEqual(result["num_violations"], 1)
        self.assertAlmostEqual(result["max_violation"], 0.3)

    def test_sim5_uses_raw_pairwise_bounds(self):
        result = self.sim5.bound_diagnostics([0.1, 0.1], [0.05, 0.3], 0.5, 2.0)
        self.assertEqual(result["num_violations"], 1)
        self.assertAlmostEqual(result["max_violation"], 0.1)


class TableGenerationTests(unittest.TestCase):
    def test_table_requires_test_split_and_writes_latex(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "summary.csv"
            output = Path(directory) / "table.tex"
            source.write_text("dataset_name,model,delta_hat,split\nisic2017,unet,0.1,test\n", encoding="utf-8")
            build_table(source, output, ("dataset_name", "model", "delta_hat"))
            self.assertIn(r"isic2017 & unet & 0.1", output.read_text(encoding="utf-8"))

    def test_table_rejects_validation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "summary.csv"
            source.write_text("dataset_name,model,delta_hat,split\nisic2017,unet,0.1,val\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-evaluation"):
                build_table(source, Path(directory) / "table.tex", ("dataset_name", "model", "delta_hat"))

    def test_multilabel_table_requires_present_macro_convention(self):
        columns = ("dataset_name", "model", "delta_macro_hat", "delta_micro_hat")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "summary.csv"
            output = Path(directory) / "table.tex"
            source.write_text(
                "dataset_name,model,delta_macro_hat,delta_micro_hat,split\n"
                "voc2012,unet,0.1,0.01,test\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "macro-class convention"):
                build_table(source, output, columns)
            source.write_text(
                "dataset_name,model,delta_macro_hat,delta_micro_hat,split,macro_class_convention\n"
                "voc2012,unet,0.1,0.01,test,fixed_all_k\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                build_table(source, output, columns)
            source.write_text(
                "dataset_name,model,delta_macro_hat,delta_micro_hat,split,macro_class_convention\n"
                f"voc2012,unet,0.1,0.01,test,{MACRO_CLASS_CONVENTION}\n",
                encoding="utf-8",
            )
            build_table(source, output, columns)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
