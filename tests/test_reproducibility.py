from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import torch

from metrics import dice_coeff, iou
from application.multilabel_segmentation.metrics import RunningMultilabelMetrics
from scripts.build_paper_tables import build_table
from scripts.check_paper_results import check_paper_results
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

    def test_readme_simulation_commands_exist(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        commands = set(re.findall(r"python (simulation/sim-[0-9-]+\.py)", readme))
        self.assertEqual(
            commands,
            {
                "simulation/sim-1-1.py",
                "simulation/sim-1-2.py",
                "simulation/sim-2.py",
                "simulation/sim-3.py",
                "simulation/sim-4.py",
                "simulation/sim-5.py",
            },
        )
        for command in commands:
            self.assertTrue((ROOT / command).is_file(), command)

    def test_cache_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "result.csv"
            csv_path.write_text("value\n1\n", encoding="utf-8")
            write_cache_metadata(csv_path, {"seed": 42})
            self.assertEqual(metadata_path(csv_path).read_text(encoding="utf-8"), '{\n  "seed": 42\n}\n')

    def test_paper_config_drives_dataset_roots(self):
        self.assertEqual(paper_config()["seed"], 42)
        self.assertEqual(dataset_config("cityscapes")["data_root"], "datasets/Cityscapes")

    def test_isic_paper_config_uses_paper_resolution_and_batch_size(self):
        config = dataset_config("isic2017")
        self.assertEqual(config["image_size"], 256)
        self.assertEqual(config["batch_size"], 24)


class MetricConventionTests(unittest.TestCase):
    def test_empty_denominator_is_zero(self):
        empty = torch.zeros((2, 1, 3, 3))
        self.assertTrue(torch.equal(dice_coeff(empty, empty), torch.zeros(2)))
        self.assertTrue(torch.equal(iou(empty, empty), torch.zeros(2)))

    def test_semantic_macro_average_uses_present_classes_only(self):
        metrics = RunningMultilabelMetrics(num_classes=3)
        targets = torch.tensor([[[0, 1]]])
        valid = torch.ones_like(targets, dtype=torch.bool)
        logits = torch.tensor([[[[20.0, -20.0]], [[-20.0, 20.0]], [[20.0, 20.0]]]])
        metrics.update(logits, targets, valid)
        result = metrics.compute_rounded(10)
        self.assertEqual(result["macro_dice"], 1.0)
        self.assertEqual(result["macro_iou"], 1.0)


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

    def test_paper_result_check_accepts_displayed_values(self):
        from scripts.check_paper_results import (
            BINARY_DELTA,
            BINARY_PERFORMANCE,
            MULTILABEL_DELTA,
            MULTILABEL_PERFORMANCE,
        )

        summaries = {
            "binary_delta": [
                {"dataset_name": dataset, "model": model, "delta_hat": value}
                for (dataset, model), value in BINARY_DELTA.items()
            ],
            "binary_performance": [
                {"dataset_name": dataset, "model": model, "optimizer": optimizer, "dice": values[0], "iou": values[1]}
                for (dataset, model), values in BINARY_PERFORMANCE.items()
                for optimizer in ("RankDice", "RankIoU")
            ],
            "multilabel_delta": [
                {"dataset_name": dataset, "model": model, "delta_macro_hat": values[0], "delta_micro_hat": values[1]}
                for (dataset, model), values in MULTILABEL_DELTA.items()
            ],
            "multilabel_performance": [
                {
                    "dataset_name": dataset,
                    "model": model,
                    "optimizer": optimizer,
                    f"{scope}_dice": values[0],
                    f"{scope}_iou": values[1],
                }
                for (dataset, model, scope), values in MULTILABEL_PERFORMANCE.items()
                for optimizer in (f"{scope}-Dice", f"{scope}-IoU")
            ],
        }
        self.assertEqual(check_paper_results(summaries), [])


if __name__ == "__main__":
    unittest.main()
