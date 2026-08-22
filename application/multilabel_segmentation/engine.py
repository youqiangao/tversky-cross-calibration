from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tversky_cross_calibration.config import paper_config
from tversky_cross_calibration.reproducibility import dataloader_generator, seed_worker
from application.checkpoints import extract_model_state_dict, infer_dataset_and_model, is_model_state_dict

from .dataset import build_dataset, default_training_config_for, normalize_dataset_name
from .metrics import RunningMultilabelMetrics, masked_index_bce_with_logits
from .models import build_model
from .utils import ensure_dir, load_checkpoint, save_checkpoint, save_json


def create_dataloader(dataset_name, data_root, split, image_size, batch_size, num_workers, shuffle, augment, max_samples=None):
    dataset = build_dataset(dataset_name, data_root, split, image_size, augment, True, max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=dataloader_generator(),
        worker_init_fn=seed_worker,
    )


def _use_amp(device: torch.device, enabled: bool) -> bool:
    return bool(enabled and device.type == "cuda")


def _autocast(device: torch.device, enabled: bool):
    return torch.cuda.amp.autocast() if _use_amp(device, enabled) else nullcontext()


def _run_epoch(model, loader, optimizer, device, threshold, scaler, amp_enabled, training: bool, compute_metrics: bool = False) -> Dict[str, float]:
    model.train(training)
    metrics = RunningMultilabelMetrics(loader.dataset.num_classes, threshold) if compute_metrics else None
    running_loss = 0.0
    progress = tqdm(loader, desc="train" if training else "eval", leave=False)
    context = nullcontext() if training else torch.no_grad()
    with context:
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp_enabled):
                logits = model(images)
                loss = masked_index_bce_with_logits(logits, masks, valid_mask)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            running_loss += float(loss.item()) * images.shape[0]
            if metrics is not None:
                metrics.update(logits, masks, valid_mask)
            progress.set_postfix(loss=f"{loss.item():.4f}")
    result = metrics.compute_rounded() if metrics is not None else {}
    result["loss"] = round(running_loss / max(len(loader.dataset), 1), 6)
    return result


def train_model(
    model_name, dataset_name, data_root, output_dir, image_size, batch_size, epochs, lr, weight_decay,
    num_workers, device, threshold=0.5, max_train_samples=None, max_val_samples=None,
    resume_checkpoint=None, amp_enabled=False,
) -> Path:
    dataset_name = normalize_dataset_name(dataset_name)
    train_loader = create_dataloader(dataset_name, data_root, "train", image_size, batch_size, num_workers, True, True, max_train_samples)
    val_loader = create_dataloader(dataset_name, data_root, "val", image_size, batch_size, num_workers, False, False, max_val_samples)
    model = build_model(model_name, train_loader.dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=_use_amp(device, amp_enabled))
    run_dir = ensure_dir(Path(output_dir) / dataset_name / model_name.lower())
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    history_path = run_dir / "history.json"
    history: List[Dict] = []
    start_epoch, best_loss = 1, float("inf")
    if resume_checkpoint is None and last_path.exists():
        existing = load_checkpoint(last_path, map_location="cpu")
        if existing.get("task_type") == "multilabel_segmentation" and int(existing.get("epoch", 0)) < int(epochs):
            resume_checkpoint = last_path
    if resume_checkpoint:
        checkpoint = load_checkpoint(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        if best_path.exists():
            best_loss = float(load_checkpoint(best_path)["val_metrics"]["loss"])

    for epoch in range(start_epoch, int(epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, threshold, scaler, amp_enabled, True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, threshold, scaler, amp_enabled, False)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        checkpoint = {
            "task_type": "multilabel_segmentation", "model_name": model_name.lower(), "dataset_name": dataset_name,
            "num_classes": train_loader.dataset.num_classes, "ignore_index": train_loader.dataset.ignore_index,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
            "image_size": image_size, "batch_size": batch_size, "threshold": threshold,
            "amp_enabled": _use_amp(device, amp_enabled), "val_metrics": val_metrics,
            "seed": 42, "precision": "fp32" if not _use_amp(device, amp_enabled) else "amp",
            "deterministic": bool(paper_config()["training"]["deterministic"]),
        }
        save_checkpoint(checkpoint, last_path)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(checkpoint, best_path)
        save_json({"history": history}, history_path)
    return best_path


def load_trained_model(checkpoint_path, device, dataset_name=None, model_name=None):
    payload = load_checkpoint(checkpoint_path, map_location=device)
    if is_model_state_dict(payload):
        inferred_dataset, inferred_model = infer_dataset_and_model(checkpoint_path)
        resolved_dataset = normalize_dataset_name(dataset_name or inferred_dataset)
        resolved_model = (model_name or inferred_model).lower()
        num_classes = {"voc2012": 21, "cityscapes": 19}[resolved_dataset]
        checkpoint = {
            "task_type": "multilabel_segmentation",
            "model_name": resolved_model,
            "dataset_name": resolved_dataset,
            "num_classes": num_classes,
            "ignore_index": 255,
            "model_state_dict": payload,
            "image_size": int(default_training_config_for(resolved_dataset)["image_size"]),
            "threshold": 0.5,
        }
    else:
        checkpoint = payload
    if checkpoint.get("task_type") != "multilabel_segmentation":
        raise ValueError(f"Expected multilabel_segmentation checkpoint, got {checkpoint.get('task_type')!r}.")
    model = build_model(checkpoint["model_name"], int(checkpoint["num_classes"])).to(device).float()
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(checkpoint_path, data_root, split, batch_size, num_workers, device, output_dir=None, dataset_name=None, max_samples=None):
    model, checkpoint = load_trained_model(checkpoint_path, device, dataset_name=dataset_name)
    dataset_name = normalize_dataset_name(dataset_name or checkpoint["dataset_name"])
    loader = create_dataloader(dataset_name, data_root, split, checkpoint["image_size"], batch_size, num_workers, False, False, max_samples)
    metrics = _run_epoch(
        model,
        loader,
        None,
        device,
        checkpoint.get("threshold", 0.5),
        None,
        False,
        False,
        compute_metrics=True,
    )
    if output_dir:
        save_json({"checkpoint": str(checkpoint_path), "split": split, "metrics": metrics}, ensure_dir(output_dir) / f"{split}_metrics.json")
    return metrics
