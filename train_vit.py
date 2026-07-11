from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import INDataset, build_samples
from models import TwoStreamVit
from utils.seed import seed_everything
from utils.train_eval import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Five-fold C/G two-stream ViT training.")
    parser.add_argument("--data-dir", default="data_content/IN_original")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--method", choices=["vit_direct", "vit_two_stage", "both"], default="both")
    parser.add_argument("--backbone", default="deit_tiny_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=0, help="GPU id to use when CUDA is available.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unshared-backbone", action="store_true")
    return parser.parse_args()


def build_transforms(image_size: int = 224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, test_tf


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def compute_metrics(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    specificities = []
    sensitivities = []
    for idx, label in enumerate(labels):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
        specificities.append(specificity)
        sensitivities.append(sensitivity)
        per_class[str(label)] = {
            "precision": precision,
            "recall": sensitivity,
            "f1": f1,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "support": int(cm[idx, :].sum()),
        }

    specificity = float(np.mean(specificities))
    sensitivity = float(np.mean(sensitivities))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def make_loader(samples, transform, batch_size, num_workers, shuffle):
    dataset = INDataset(samples, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def predict(model, loader, device):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["c"].to(device), batch["g"].to(device))
            y_true.extend(batch["label"].tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            sample_ids.extend(batch["sample_id"])
    return y_true, y_pred, sample_ids


def save_predictions(path: Path, sample_ids: list[str], y_true: list[int], y_pred: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "pred"])
        writer.writerows(zip(sample_ids, y_true, y_pred))


def train_direct(args, fold, train_samples, test_samples, train_tf, test_tf, device, method_root: Path):
    fold_dir = method_root / f"fold_{fold}"
    train_loader = make_loader(train_samples, train_tf, args.batch_size, args.num_workers, True)
    test_loader = make_loader(test_samples, test_tf, args.batch_size, args.num_workers, False)
    model = TwoStreamVit(
        backbone=args.backbone,
        num_classes=3,
        pretrained=not args.no_pretrained,
        share_backbone=not args.unshared_backbone,
    ).to(device)
    train_labels = [s.label for s in train_samples]
    ckpt = fold_dir / "best.pt"
    model, best = train_model(
        model,
        train_loader,
        test_loader,
        train_labels,
        3,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        ckpt,
    )
    y_true, y_pred, sample_ids = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, labels=[0, 1, 2])
    metrics["best"] = best
    save_predictions(fold_dir / "test_predictions.csv", sample_ids, y_true, y_pred)
    return metrics


def relabel(samples, fn):
    copied = []
    for s in samples:
        copied.append(type(s)(sample_id=s.sample_id, label=fn(s.label), c_path=s.c_path, g_path=s.g_path))
    return copied


def train_binary(args, name, fold_dir, train_samples, test_samples, train_tf, test_tf, device):
    train_loader = make_loader(train_samples, train_tf, args.batch_size, args.num_workers, True)
    test_loader = make_loader(test_samples, test_tf, args.batch_size, args.num_workers, False)
    model = TwoStreamVit(
        backbone=args.backbone,
        num_classes=2,
        pretrained=not args.no_pretrained,
        share_backbone=not args.unshared_backbone,
    ).to(device)
    ckpt = fold_dir / f"{name}.pt"
    model, best = train_model(
        model,
        train_loader,
        test_loader,
        [s.label for s in train_samples],
        2,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        ckpt,
    )
    return model, best


def train_two_stage(args, fold, train_samples, test_samples, train_tf, test_tf, device, method_root: Path):
    fold_dir = method_root / f"fold_{fold}"
    stage1_train = relabel(train_samples, lambda y: 0 if y == 0 else 1)
    stage1_test = relabel(test_samples, lambda y: 0 if y == 0 else 1)
    stage1_model, stage1_best = train_binary(
        args, "stage1_0_vs_12", fold_dir, stage1_train, stage1_test, train_tf, test_tf, device
    )

    stage2_train_raw = [s for s in train_samples if s.label in {1, 2}]
    stage2_test_raw = [s for s in test_samples if s.label in {1, 2}]
    stage2_train = relabel(stage2_train_raw, lambda y: 0 if y == 1 else 1)
    stage2_test = relabel(stage2_test_raw, lambda y: 0 if y == 1 else 1)
    stage2_model, stage2_best = train_binary(
        args, "stage2_1_vs_2", fold_dir, stage2_train, stage2_test, train_tf, test_tf, device
    )

    test_loader = make_loader(test_samples, test_tf, args.batch_size, args.num_workers, False)
    stage1_model.eval()
    stage2_model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in test_loader:
            c_img = batch["c"].to(device)
            g_img = batch["g"].to(device)
            first = stage1_model(c_img, g_img).argmax(dim=1)
            second = stage2_model(c_img, g_img).argmax(dim=1)
            pred = torch.where(first == 0, torch.zeros_like(first), second + 1)
            y_true.extend(batch["label"].tolist())
            y_pred.extend(pred.cpu().tolist())
            sample_ids.extend(batch["sample_id"])
    metrics = compute_metrics(y_true, y_pred, labels=[0, 1, 2])
    metrics["stage1_best"] = stage1_best
    metrics["stage2_best"] = stage2_best
    save_predictions(fold_dir / "test_predictions.csv", sample_ids, y_true, y_pred)
    return metrics


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    seed_everything(args.seed)
    samples = build_samples(args.data_dir)
    labels = np.asarray([s.label for s in samples])
    print(f"samples={len(samples)} label_counts={dict(Counter(labels.tolist()))}")
    print(f"default backbone={args.backbone}")
    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(f"--gpu {args.gpu} was requested, but only {torch.cuda.device_count()} CUDA device(s) found.")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"device={device}")
    train_tf, test_tf = build_transforms(args.image_size)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_results: dict[str, list[dict]] = {"vit_direct": [], "vit_two_stage": []}
    method_roots = {
        "vit_direct": Path(args.output_dir) / "vit_direct" / timestamp,
        "vit_two_stage": Path(args.output_dir) / "vit_two_stage" / timestamp,
    }
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n===== fold {fold}/{args.folds} =====")
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]
        if args.method in {"vit_direct", "both"}:
            metrics = train_direct(
                args, fold, train_samples, test_samples, train_tf, test_tf, device, method_roots["vit_direct"]
            )
            all_results["vit_direct"].append(metrics)
            save_json(method_roots["vit_direct"] / f"fold_{fold}" / "test_metrics.json", metrics)
        if args.method in {"vit_two_stage", "both"}:
            metrics = train_two_stage(
                args, fold, train_samples, test_samples, train_tf, test_tf, device, method_roots["vit_two_stage"]
            )
            all_results["vit_two_stage"].append(metrics)
            save_json(method_roots["vit_two_stage"] / f"fold_{fold}" / "test_metrics.json", metrics)

    summary = {}
    for method, rows in all_results.items():
        if not rows:
            continue
        summary[method] = {
            "output_dir": str(method_roots[method]),
            "acc": mean_std([r["acc"] for r in rows]),
            "bacc": mean_std([r["bacc"] for r in rows]),
            "precision": mean_std([r["precision"] for r in rows]),
            "recall": mean_std([r["recall"] for r in rows]),
            "f1": mean_std([r["f1"] for r in rows]),
            "specificity": mean_std([r["specificity"] for r in rows]),
            "sensitivity": mean_std([r["sensitivity"] for r in rows]),
        }
    save_json(Path(args.output_dir) / f"summary_{timestamp}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
