"""Download Kvasir-SEG and create the paper's deterministic 720/80/200 split."""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import zipfile
from pathlib import Path


KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_kvasir(data_root: Path, seed: int = 42) -> Path:
    downloads = ensure_dir(data_root / "downloads")
    archive_path = downloads / "kvasir-seg.zip"
    if not archive_path.exists():
        subprocess.run(
            ["curl", "-L", "--fail", "--retry", "8", "-o", str(archive_path), KVASIR_URL],
            check=True,
        )
    extract_root = downloads / "kvasir-seg"
    if not extract_root.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_root)

    source = extract_root / "Kvasir-SEG"
    images = source / "images"
    masks = source / "masks"
    image_ids = [path.stem for path in sorted(images.glob("*.jpg"))]
    if len(image_ids) != 1000:
        raise RuntimeError(f"Expected 1000 Kvasir images, found {len(image_ids)}.")
    random.Random(seed).shuffle(image_ids)
    split_ids = {
        "test": image_ids[:200],
        "val": image_ids[200:280],
        "train": image_ids[280:],
    }

    target = data_root / "Kvasir-SEG" / "paper_split"
    rows = ["image_id,split\n"]
    for split, ids in split_ids.items():
        image_dir = ensure_dir(target / split / "images")
        mask_dir = ensure_dir(target / split / "masks")
        for image_id in ids:
            shutil.copy2(images / f"{image_id}.jpg", image_dir / f"{image_id}.jpg")
            shutil.copy2(masks / f"{image_id}.jpg", mask_dir / f"{image_id}.jpg")
            rows.append(f"{image_id},{split}\n")
    (target / "manifest.csv").write_text("".join(rows), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare_kvasir(args.data_root, args.seed))


if __name__ == "__main__":
    main()
