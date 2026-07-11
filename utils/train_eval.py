from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def class_weight_tensor(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / max(counts.get(cls, 0), 1) for cls in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights / weights.mean()


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, leave=False):
            c_img = batch["c"].to(device, non_blocking=True)
            g_img = batch["g"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(c_img, g_img)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    avg_loss = loss_sum / max(len(loader.dataset), 1)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    acc = correct / max(len(y_true), 1)
    return avg_loss, acc, y_true, y_pred


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: list[int],
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    save_path: Path,
):
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(train_labels, num_classes, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = {"balanced_acc": -1.0, "epoch": 0}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, _ = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, y_true, y_pred = run_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        per_class_recall = []
        for cls in range(num_classes):
            total = sum(1 for y in y_true if y == cls)
            hit = sum(1 for y, p in zip(y_true, y_pred) if y == cls and p == cls)
            per_class_recall.append(hit / total if total else 0.0)
        balanced_acc = sum(per_class_recall) / num_classes
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"eval_loss={val_loss:.4f} eval_acc={val_acc:.4f} eval_bacc={balanced_acc:.4f}"
        )
        if balanced_acc > best["balanced_acc"]:
            best = {"balanced_acc": balanced_acc, "epoch": epoch}
            torch.save({"model": model.state_dict(), "best": best}, save_path)

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model, best
