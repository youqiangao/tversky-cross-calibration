"""Prepare or validate the five paper datasets from local official archives."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import shutil
import stat
import tarfile
import zipfile


DATASETS = ("oxford_pet", "isic2017", "kvasir_seg", "voc2012", "cityscapes")
EXPECTED_COUNTS = {
    "oxford_pet": {"images": 7390, "masks": 7390},
    "isic2017": {"train": 2000, "val": 150, "test": 600, "masks": 2750},
    "kvasir_seg": {"train": 720, "val": 80, "test": 200},
    "voc2012": {"train": 1464, "val": 1449},
    "cityscapes": {"train": 2975, "val": 500},
}
OXFORD_FILES = ("images.tar.gz", "annotations.tar.gz")
ISIC_FILES = (
    "ISIC-2017_Training_Data.zip",
    "ISIC-2017_Training_Part1_GroundTruth.zip",
    "ISIC-2017_Validation_Data.zip",
    "ISIC-2017_Validation_Part1_GroundTruth.zip",
    "ISIC-2017_Test_v2_Data.zip",
    "ISIC-2017_Test_v2_Part1_GroundTruth.zip",
)
KVASIR_FILE = "kvasir-seg.zip"
VOC_FILE = "VOCtrainval_11-May-2012.tar"
CITYSCAPES_FILES = ("leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip")


def _safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    if root.resolve() not in (target, *target.parents):
        raise RuntimeError(f"Archive member escapes destination: {member_name}")
    return target


def extract_tar(path: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            _safe_target(root, member.name)
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not supported: {member.name}")
        archive.extractall(root)


def extract_zip(path: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            _safe_target(root, member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"Archive links are not supported: {member.filename}")
        archive.extractall(root)


def _archive(download_dir: Path, dataset: str, filename: str) -> Path:
    path = download_dir / dataset / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing official archive: {path}. Download {filename} to that directory, then rerun this command."
        )
    if path.stat().st_size == 0:
        raise RuntimeError(f"Archive is empty: {path}")
    return path


def prepare_oxford(data_root: Path, download_dir: Path | None = None) -> Path:
    download_dir = download_dir or data_root / "downloads"
    target = data_root / "OxfordIIITPet"
    for filename in OXFORD_FILES:
        expected = target / ("images" if filename.startswith("images") else "annotations")
        if not expected.exists():
            extract_tar(_archive(download_dir, "oxford_pet", filename), target)
    return target


def prepare_isic(data_root: Path, download_dir: Path | None = None) -> Path:
    download_dir = download_dir or data_root / "downloads"
    target = data_root / "ISIC-2017"
    for filename in ISIC_FILES:
        folder = target / filename.removesuffix(".zip")
        if not folder.exists():
            extract_zip(_archive(download_dir, "isic2017", filename), target)
    for path in target.glob("*_Part1_GroundTruth/*_Segmentation.png"):
        normalized = path.with_name(path.name.replace("_Segmentation.png", "_segmentation.png"))
        if not normalized.exists():
            path.rename(normalized)
    return target


def _kvasir_manifest_is_complete(target: Path) -> bool:
    manifest = target / "manifest.csv"
    if not manifest.is_file():
        return False
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1000 or len({row.get("image_id") for row in rows}) != 1000:
        return False
    expected = EXPECTED_COUNTS["kvasir_seg"]
    if not all(sum(row.get("split") == split for row in rows) == count for split, count in expected.items()):
        return False
    return all(
        (target / str(row["split"]) / "images" / f"{row['image_id']}.jpg").is_file()
        and (target / str(row["split"]) / "masks" / f"{row['image_id']}.jpg").is_file()
        for row in rows
    )


def prepare_kvasir(data_root: Path, download_dir: Path | None = None, seed: int = 42) -> Path:
    download_dir = download_dir or data_root / "downloads"
    target = data_root / "Kvasir-SEG" / "paper_split"
    if _kvasir_manifest_is_complete(target):
        return target

    extract_root = download_dir / "kvasir_seg" / "extracted"
    source = extract_root / "Kvasir-SEG"
    if not source.exists():
        extract_zip(_archive(download_dir, "kvasir_seg", KVASIR_FILE), extract_root)
    images = source / "images"
    masks = source / "masks"
    image_ids = [path.stem for path in sorted(images.glob("*.jpg"))]
    if len(image_ids) != 1000:
        raise RuntimeError(f"Expected 1000 Kvasir images in {images}, found {len(image_ids)}.")
    missing_masks = [image_id for image_id in image_ids if not (masks / f"{image_id}.jpg").is_file()]
    if missing_masks:
        raise RuntimeError(f"Kvasir-SEG has {len(missing_masks)} images without masks.")

    random.Random(seed).shuffle(image_ids)
    split_ids = {"test": image_ids[:200], "val": image_ids[200:280], "train": image_ids[280:]}
    rows = ["image_id,split\n"]
    for split, ids in split_ids.items():
        image_dir = target / split / "images"
        mask_dir = target / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            shutil.copy2(images / f"{image_id}.jpg", image_dir / f"{image_id}.jpg")
            shutil.copy2(masks / f"{image_id}.jpg", mask_dir / f"{image_id}.jpg")
            rows.append(f"{image_id},{split}\n")
    (target / "manifest.csv").write_text("".join(rows), encoding="utf-8")
    return target


def prepare_voc(data_root: Path, download_dir: Path | None = None) -> Path:
    download_dir = download_dir or data_root / "downloads"
    target = data_root / "VOCtrainval_11-May-2012"
    expected = target / "VOCdevkit" / "VOC2012"
    if not expected.exists():
        extract_tar(_archive(download_dir, "voc2012", VOC_FILE), target)
    return target


def prepare_cityscapes(data_root: Path, download_dir: Path | None = None) -> Path:
    download_dir = download_dir or data_root / "downloads"
    target = data_root / "Cityscapes"
    expected = (target / "leftImg8bit", target / "gtFine")
    for filename, folder in zip(CITYSCAPES_FILES, expected):
        if not folder.exists():
            extract_zip(_archive(download_dir, "cityscapes", filename), target)
    return target


def validate(dataset: str, data_root: Path) -> dict[str, int]:
    if dataset == "oxford_pet":
        root = data_root / "OxfordIIITPet"
        images = [path for path in (root / "images").glob("*.jpg") if not path.name.startswith(".")]
        masks = [path for path in (root / "annotations" / "trimaps").glob("*.png") if not path.name.startswith(".")]
        counts = {"images": len(images), "masks": len(masks)}
        missing = [path.stem for path in images if not (root / "annotations" / "trimaps" / f"{path.stem}.png").exists()]
    elif dataset == "isic2017":
        root = data_root / "ISIC-2017"
        counts = {
            split: len(list((root / folder).glob("ISIC_*.jpg")))
            for split, folder in (("train", "ISIC-2017_Training_Data"), ("val", "ISIC-2017_Validation_Data"), ("test", "ISIC-2017_Test_v2_Data"))
        }
        counts["masks"] = sum(len(list(folder.glob("*_segmentation.png"))) for folder in root.glob("*_Part1_GroundTruth"))
        missing = []
        for image_folder in root.glob("*_Data"):
            mask_folder = root / image_folder.name.replace("_Data", "_Part1_GroundTruth")
            missing.extend(path.stem for path in image_folder.glob("ISIC_*.jpg") if not (mask_folder / f"{path.stem}_segmentation.png").exists())
    elif dataset == "kvasir_seg":
        root = data_root / "Kvasir-SEG" / "paper_split"
        counts = {split: len(list((root / split / "images").glob("*.jpg"))) for split in ("train", "val", "test")}
        missing = []
        for split in ("train", "val", "test"):
            missing.extend(path.stem for path in (root / split / "images").glob("*.jpg") if not (root / split / "masks" / path.name).exists())
        if not _kvasir_manifest_is_complete(root):
            missing.append("manifest.csv")
    elif dataset == "voc2012":
        root = data_root / "VOCtrainval_11-May-2012" / "VOCdevkit" / "VOC2012"
        ids = {
            split: [line.strip() for line in (root / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().splitlines() if line.strip()]
            for split in ("train", "val")
        }
        counts = {split: len(values) for split, values in ids.items()}
        missing = [
            image_id for values in ids.values() for image_id in values
            if not (root / "JPEGImages" / f"{image_id}.jpg").exists()
            or not (root / "SegmentationClass" / f"{image_id}.png").exists()
        ]
    else:
        root = data_root / "Cityscapes"
        counts = {split: len(list((root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))) for split in ("train", "val")}
        missing = []
        for split in ("train", "val"):
            for path in (root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"):
                image_id = path.name.removesuffix("_leftImg8bit.png")
                mask = root / "gtFine" / split / path.parent.name / f"{image_id}_gtFine_labelIds.png"
                if not mask.exists():
                    missing.append(image_id)
    if counts != EXPECTED_COUNTS[dataset] or missing:
        detail = f"; {len(missing)} required files or pairs are incomplete" if missing else ""
        raise RuntimeError(f"Incomplete {dataset} dataset: {counts}; expected {EXPECTED_COUNTS[dataset]}{detail}")
    return counts


def prepare(dataset: str, data_root: Path, download_dir: Path | None = None) -> Path:
    functions = {
        "oxford_pet": prepare_oxford,
        "isic2017": prepare_isic,
        "kvasir_seg": prepare_kvasir,
        "voc2012": prepare_voc,
        "cityscapes": prepare_cityscapes,
    }
    return functions[dataset](data_root, download_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASETS, default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if args.all else tuple(args.dataset)
    if not datasets:
        parser.error("select --all or at least one --dataset")
    download_dir = args.download_dir or args.data_root / "downloads"
    for dataset in datasets:
        if not args.check:
            print(f"prepared {dataset}: {prepare(dataset, args.data_root, download_dir)}")
        print(f"checked {dataset}: {validate(dataset, args.data_root)}")


if __name__ == "__main__":
    main()
