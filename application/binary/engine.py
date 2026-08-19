from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tversky_cross_calibration.config import paper_config
from tversky_cross_calibration.reproducibility import dataloader_generator, seed_worker

from .dataset import build_dataset, normalize_dataset_name
from .metrics import RunningSegmentationMetrics
from .models import build_model
from .utils import ensure_dir, load_checkpoint, save_checkpoint, save_json


def json_load(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def create_dataloader(
    dataset_name: str,
    data_root: str | Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: bool,
    val_split: float = 0.1,
    split_seed: int = 42,
    max_samples: int | None = None,
) -> DataLoader:
    dataset = build_dataset(dataset_name, data_root, split, image_size, augment, True, max_samples, val_split, split_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=dataloader_generator(),
        worker_init_fn=seed_worker,
    )


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _dataset_config(dataset) -> Tuple[int, int]:
    return int(getattr(dataset, "num_classes", 2)), int(getattr(dataset, "ignore_index", 255))


def train_one_epoch(model, loader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    metrics = RunningSegmentationMetrics(ignore_index=getattr(criterion, "ignore_index", 255))
    running_loss = 0.0
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        metrics.update(logits.detach(), masks)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    result = metrics.compute_rounded()
    result["loss"] = round(running_loss / max(len(loader.dataset), 1), 6)
    return result


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    metrics = RunningSegmentationMetrics(ignore_index=getattr(criterion, "ignore_index", 255))
    running_loss = 0.0
    progress = tqdm(loader, desc="eval", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)
        metrics.update(logits, masks)
    result = metrics.compute_rounded()
    result["loss"] = round(running_loss / max(len(loader.dataset), 1), 6)
    return result


def train_model(
    model_name: str,
    dataset_name: str,
    data_root: str | Path,
    output_dir: str | Path,
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    device: torch.device,
    val_split: float = 0.1,
    split_seed: int = 42,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    dataset_name = normalize_dataset_name(dataset_name)
    train_loader = create_dataloader(dataset_name, data_root, "train", image_size, batch_size, num_workers, True, True, val_split, split_seed, max_train_samples)
    val_loader = create_dataloader(dataset_name, data_root, "val", image_size, batch_size, num_workers, False, False, val_split, split_seed, max_val_samples)
    num_classes, ignore_index = _dataset_config(train_loader.dataset)
    model = build_model(model_name, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = build_optimizer(model, lr, weight_decay)

    run_dir = ensure_dir(Path(output_dir) / dataset_name / model_name.lower())
    best_checkpoint_path = run_dir / "best.pt"
    history: List[Dict[str, float]] = []
    best_val_ce = float("inf")
    start_epoch = 1

    history_path = run_dir / "history.json"
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        checkpoint = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if history_path.exists():
            history_payload = json_load(history_path)
            history = list(history_payload.get("history", []))
        if best_checkpoint_path.exists():
            best_checkpoint = load_checkpoint(best_checkpoint_path, map_location="cpu")
            best_val_ce = float(best_checkpoint.get("val_metrics", {}).get("loss", float("inf")))
        elif checkpoint.get("val_metrics") is not None:
            best_val_ce = float(checkpoint["val_metrics"].get("loss", float("inf")))

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        checkpoint = {
            "task_type": "binary",
            "model_name": model_name.lower(),
            "dataset_name": dataset_name,
            "num_classes": int(num_classes),
            "ignore_index": int(ignore_index),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "image_size": image_size,
            "val_split": float(val_split),
            "split_seed": int(split_seed),
            "val_metrics": val_metrics,
            "seed": int(split_seed),
            "precision": "fp32",
            "deterministic": bool(paper_config()["training"]["deterministic"]),
        }
        save_checkpoint(checkpoint, run_dir / "last.pt")
        if val_metrics["loss"] < best_val_ce:
            best_val_ce = val_metrics["loss"]
            save_checkpoint(checkpoint, best_checkpoint_path)

    save_json({"history": history}, run_dir / "history.json")
    return best_checkpoint_path


def load_trained_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_name"], num_classes=int(checkpoint.get("num_classes", 2))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
