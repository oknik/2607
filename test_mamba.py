from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from models import TwoStreamMamba
from test_vit import (
    build_test_transform,
    compute_metrics,
    fold_indices,
    get_device,
    make_loader,
    mean_std,
    save_json,
)
from datasets import build_samples
from utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone test for trained C/G two-stream Mamba checkpoints.")
    parser.add_argument("--data-dir", default="data_content/IN_original")
    parser.add_argument("--checkpoint-dir", required=True, help="Example: outputs/mamba_direct/20260713_153000")
    parser.add_argument("--output-dir", default=None, help="Default: <checkpoint-dir>/standalone_test_<timestamp>")
    parser.add_argument("--method", choices=["mamba_direct", "mamba_two_stage"], required=True)
    parser.add_argument("--fold", default="all", help="Use all or a fold index such as 1.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="vim_tiny_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--unshared-backbone", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, num_classes: int, args, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = TwoStreamMamba(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=False,
        share_backbone=not args.unshared_backbone,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_direct(model, loader, device):
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["c"].to(device), batch["g"].to(device))
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            for sample_id, label, pred, logit, prob in zip(
                batch["sample_id"], batch["label"].tolist(), preds.cpu().tolist(), logits.cpu().tolist(), probs.cpu().tolist()
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(label),
                        "pred": int(pred),
                        "logits": logit,
                        "probs": prob,
                    }
                )
    return rows


def predict_two_stage(stage1_model, stage2_model, loader, device):
    rows = []
    with torch.no_grad():
        for batch in loader:
            c_img = batch["c"].to(device)
            g_img = batch["g"].to(device)
            stage1_logits = stage1_model(c_img, g_img)
            stage2_logits = stage2_model(c_img, g_img)
            stage1_probs = torch.softmax(stage1_logits, dim=1)
            stage2_probs = torch.softmax(stage2_logits, dim=1)
            stage1_pred = stage1_probs.argmax(dim=1)
            stage2_pred = stage2_probs.argmax(dim=1)
            final_pred = torch.where(stage1_pred == 0, torch.zeros_like(stage1_pred), stage2_pred + 1)
            for sample_id, label, pred, s1_logit, s1_prob, s2_logit, s2_prob in zip(
                batch["sample_id"],
                batch["label"].tolist(),
                final_pred.cpu().tolist(),
                stage1_logits.cpu().tolist(),
                stage1_probs.cpu().tolist(),
                stage2_logits.cpu().tolist(),
                stage2_probs.cpu().tolist(),
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(label),
                        "pred": int(pred),
                        "stage1_logits": s1_logit,
                        "stage1_probs": s1_prob,
                        "stage2_logits": s2_logit,
                        "stage2_probs": s2_prob,
                    }
                )
    return rows


def save_predictions(path: Path, rows: list[dict], method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if method == "mamba_direct":
        fieldnames = ["sample_id", "label", "pred", "logit_0", "logit_1", "logit_2", "prob_0", "prob_1", "prob_2"]
    else:
        fieldnames = [
            "sample_id",
            "label",
            "pred",
            "stage1_logit_0",
            "stage1_logit_1",
            "stage1_prob_0",
            "stage1_prob_1",
            "stage2_logit_1",
            "stage2_logit_2",
            "stage2_prob_1",
            "stage2_prob_2",
        ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if method == "mamba_direct":
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "pred": row["pred"],
                        "logit_0": row["logits"][0],
                        "logit_1": row["logits"][1],
                        "logit_2": row["logits"][2],
                        "prob_0": row["probs"][0],
                        "prob_1": row["probs"][1],
                        "prob_2": row["probs"][2],
                    }
                )
            else:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "pred": row["pred"],
                        "stage1_logit_0": row["stage1_logits"][0],
                        "stage1_logit_1": row["stage1_logits"][1],
                        "stage1_prob_0": row["stage1_probs"][0],
                        "stage1_prob_1": row["stage1_probs"][1],
                        "stage2_logit_1": row["stage2_logits"][0],
                        "stage2_logit_2": row["stage2_logits"][1],
                        "stage2_prob_1": row["stage2_probs"][0],
                        "stage2_prob_2": row["stage2_probs"][1],
                    }
                )


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.gpu)
    print(f"device={device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / f"standalone_test_{datetime.now():%Y%m%d_%H%M%S}"
    samples = build_samples(args.data_dir)
    splits = fold_indices(samples, args.folds, args.seed)
    selected_folds = range(1, args.folds + 1) if args.fold == "all" else [int(args.fold)]
    test_tf = build_test_transform(args.image_size)

    all_metrics = []
    for fold in selected_folds:
        if fold < 1 or fold > args.folds:
            raise ValueError(f"--fold must be all or an integer in [1, {args.folds}]")
        _, test_idx = splits[fold - 1]
        test_samples = [samples[i] for i in test_idx]
        loader = make_loader(test_samples, test_tf, args.batch_size, args.num_workers)
        fold_ckpt_dir = checkpoint_dir / f"fold_{fold}"

        if args.method == "mamba_direct":
            model = load_model(fold_ckpt_dir / "best.pt", 3, args, device)
            rows = predict_direct(model, loader, device)
        else:
            stage1 = load_model(fold_ckpt_dir / "stage1_0_vs_12.pt", 2, args, device)
            stage2 = load_model(fold_ckpt_dir / "stage2_1_vs_2.pt", 2, args, device)
            rows = predict_two_stage(stage1, stage2, loader, device)

        y_true = [row["label"] for row in rows]
        y_pred = [row["pred"] for row in rows]
        metrics = compute_metrics(y_true, y_pred, labels=[0, 1, 2])
        metrics["fold"] = fold
        metrics["num_samples"] = len(rows)
        all_metrics.append(metrics)

        fold_out = output_dir / f"fold_{fold}"
        save_json(fold_out / "test_metrics.json", metrics)
        save_predictions(fold_out / "test_predictions.csv", rows, args.method)
        print(f"fold={fold} acc={metrics['acc']:.4f} bacc={metrics['bacc']:.4f} f1={metrics['f1']:.4f}")

    summary = {
        "method": args.method,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "folds": list(selected_folds),
        "acc": mean_std([m["acc"] for m in all_metrics]),
        "bacc": mean_std([m["bacc"] for m in all_metrics]),
        "precision": mean_std([m["precision"] for m in all_metrics]),
        "recall": mean_std([m["recall"] for m in all_metrics]),
        "f1": mean_std([m["f1"] for m in all_metrics]),
        "specificity": mean_std([m["specificity"] for m in all_metrics]),
        "sensitivity": mean_std([m["sensitivity"] for m in all_metrics]),
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
