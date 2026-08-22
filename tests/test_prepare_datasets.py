from __future__ import annotations

import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest import mock

from application.prepare_datasets import (
    DATASETS,
    ISIC_FILES,
    _safe_target,
    extract_tar,
    prepare_isic,
    prepare_kvasir,
    prepare_oxford,
    validate,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class PrepareDatasetsTests(unittest.TestCase):
    def test_safe_target_rejects_archive_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "extract"
            root.mkdir()
            self.assertEqual(_safe_target(root, "folder/file.txt"), (root / "folder/file.txt").resolve())
            with self.assertRaises(RuntimeError):
                _safe_target(root, "../escape.txt")

    def test_tar_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "links.tar"
            with tarfile.open(archive_path, "w") as archive:
                member = tarfile.TarInfo("linked-file")
                member.type = tarfile.SYMTYPE
                member.linkname = "../outside"
                archive.addfile(member)
            with self.assertRaisesRegex(RuntimeError, "links are not supported"):
                extract_tar(archive_path, root / "extract")

    def test_validate_minimal_supported_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            touch(root / "OxfordIIITPet/images/a.jpg")
            touch(root / "OxfordIIITPet/annotations/trimaps/a.png")

            for folder in ("Training", "Validation", "Test_v2"):
                touch(root / f"ISIC-2017/ISIC-2017_{folder}_Data/ISIC_1.jpg")
                touch(root / f"ISIC-2017/ISIC-2017_{folder}_Part1_GroundTruth/ISIC_1_segmentation.png")

            for split in ("train", "val", "test"):
                touch(root / f"Kvasir-SEG/paper_split/{split}/images/a.jpg")
                touch(root / f"Kvasir-SEG/paper_split/{split}/masks/a.jpg")
            (root / "Kvasir-SEG/paper_split/manifest.csv").write_text(
                "image_id,split\na,train\n", encoding="utf-8"
            )

            voc = root / "VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation"
            voc.mkdir(parents=True)
            (voc / "train.txt").write_text("a\n", encoding="utf-8")
            (voc / "val.txt").write_text("b\n", encoding="utf-8")

            for split in ("train", "val"):
                touch(root / f"Cityscapes/leftImg8bit/{split}/city/a_leftImg8bit.png")
                touch(root / f"Cityscapes/gtFine/{split}/city/a_gtFine_labelIds.png")

            touch(root / "VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/a.jpg")
            touch(root / "VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/b.jpg")
            touch(root / "VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/a.png")
            touch(root / "VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/b.png")

            expected = {
                "oxford_pet": {"images": 1, "masks": 1},
                "isic2017": {"train": 1, "val": 1, "test": 1, "masks": 3},
                "kvasir_seg": {"train": 1, "val": 1, "test": 1},
                "voc2012": {"train": 1, "val": 1},
                "cityscapes": {"train": 1, "val": 1},
            }
            with mock.patch("application.prepare_datasets.EXPECTED_COUNTS", expected):
                # The compact fixture uses a one-row Kvasir manifest.
                with mock.patch("application.prepare_datasets._kvasir_manifest_is_complete", return_value=True):
                    for dataset in DATASETS:
                        self.assertEqual(validate(dataset, root), expected[dataset])

    def test_isic_ground_truth_names_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "ISIC-2017"
            for filename in ISIC_FILES:
                (target / filename.removesuffix(".zip")).mkdir(parents=True)
            original = target / "ISIC-2017_Training_Part1_GroundTruth/ISIC_1_Segmentation.png"
            touch(original)
            prepare_isic(root)
            self.assertFalse(original.exists())
            self.assertTrue(original.with_name("ISIC_1_segmentation.png").exists())

    def test_kvasir_split_is_fixed_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "downloads/kvasir_seg/extracted/Kvasir-SEG"
            for index in range(1000):
                touch(source / f"images/sample_{index:04d}.jpg")
                touch(source / f"masks/sample_{index:04d}.jpg")
            first = prepare_kvasir(root, seed=42)
            first_manifest = (first / "manifest.csv").read_text(encoding="utf-8")
            second = prepare_kvasir(root, seed=42)
            self.assertEqual(first_manifest, (second / "manifest.csv").read_text(encoding="utf-8"))
            self.assertEqual({split: len(list((first / split / "images").glob("*.jpg"))) for split in ("train", "val", "test")}, {"train": 720, "val": 80, "test": 200})

    def test_prepare_reports_the_expected_local_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "datasets/downloads/oxford_pet/images.tar.gz"):
                prepare_oxford(root / "datasets")


if __name__ == "__main__":
    unittest.main()
