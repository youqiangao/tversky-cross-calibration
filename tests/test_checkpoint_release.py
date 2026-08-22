from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from application.checkpoints import (
    PAPER_CHECKPOINTS,
    checkpoint_paths,
    extract_model_state_dict,
    hf_download_command,
    infer_dataset_and_model,
    is_model_state_dict,
)
from scripts.export_checkpoints import export_one


class CheckpointReleaseTests(unittest.TestCase):
    def test_paper_release_has_ten_unique_paths(self) -> None:
        paths = [checkpoint_paths(item) for item in PAPER_CHECKPOINTS]
        self.assertEqual(len(paths), 10)
        self.assertEqual(len(set(paths)), 10)
        self.assertEqual({item["model"] for item in PAPER_CHECKPOINTS}, {"unet", "fcn8"})
        self.assertEqual(
            {item["dataset"] for item in PAPER_CHECKPOINTS},
            {"oxford_pet", "isic2017", "kvasir_seg", "voc2012", "cityscapes"},
        )

    def test_export_contains_only_state_dict_tensors(self) -> None:
        state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "counter": torch.tensor(4)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            destination = root / "public.pt"
            torch.save({"model_state_dict": state, "optimizer_state_dict": {"secret": 1}, "epoch": 9}, source)
            size = export_one(source, destination)
            exported = torch.load(destination, map_location="cpu")
        self.assertTrue(is_model_state_dict(exported))
        self.assertNotIn("model_state_dict", exported)
        self.assertNotIn("optimizer_state_dict", exported)
        self.assertTrue(torch.equal(exported["weight"], state["weight"]))
        self.assertGreater(size, 0)

    def test_state_dict_detection_supports_legacy_and_public_files(self) -> None:
        state = {"weight": torch.ones(1)}
        self.assertIs(extract_model_state_dict(state), state)
        self.assertIs(extract_model_state_dict({"model_state_dict": state}), state)
        with self.assertRaises(ValueError):
            extract_model_state_dict({"epoch": 1})

    def test_infer_identity_from_canonical_path(self) -> None:
        dataset, model = infer_dataset_and_model("outputs/application/binary/isic2017/unet/best.pt")
        self.assertEqual((dataset, model), ("isic2017", "unet"))

    def test_hugging_face_command_targets_the_canonical_tree(self) -> None:
        command = hf_download_command(PAPER_CHECKPOINTS[0])
        self.assertIn("binary/oxford_pet/unet/best.pt", command)
        self.assertTrue(command.endswith("--local-dir outputs/application"))


if __name__ == "__main__":
    unittest.main()
