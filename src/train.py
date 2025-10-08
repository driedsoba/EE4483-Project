# src/train.py
"""
Generic training script for any model registered under src/models/*/model.py

Usage example:
    python -m src.train --model resnet18 --pretrained --epochs 12 --warmup_epochs 2 \
        --data_root datasets --out_dir runs

    python -m src.train --model my_sequential --epochs 30 --warmup_epochs 0 \
        --data_root datasets --out_dir runs
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import make_loaders
from src.registry import discover_models


def train_once(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data ---
    train_loader, val_loader, class_names = make_loaders(
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        aug_strength=args.aug_strength,
        seed=args.seed
    )

    # --- Model discovery ---
    builders = discover_models()
    if args.model not in builders:
        raise ValueError(f"Model '{args.model}' not found. Available: {list(builders.keys())}")
    model = builders[args.model](num_classes=len(class_names),
                                 pretrained=args.pretrained,
                                 freeze="head" if args.warmup_epochs > 0 else "none").to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_path = out_dir / "best.pt"

    def run_epoch(train_flag: bool):
        model.train(train_flag)
        loader = train_loader if train_flag else val_loader
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train_flag:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            if train_flag:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        return loss_sum / total, correct / total

    # --- Warm-up (train head only) ---
    for ep in range(args.warmup_epochs):
        tr_l, tr_a = run_epoch(True)
        va_l, va_a = run_epoch(False)
        if va_a > best_acc:
            best_acc = va_a
            torch.save(model.state_dict(), best_path)
        print(f"[{args.model}] Warmup {ep+1}/{args.warmup_epochs} "
              f"| train acc={tr_a:.4f} val acc={va_a:.4f} best={best_acc:.4f}")

    # --- Unfreeze and fine-tune ---
    if args.warmup_epochs > 0:
        # rebuild full model & load warmup weights
        model = builders[args.model](num_classes=len(class_names),
                                     pretrained=args.pretrained,
                                     freeze="none").to(device)
        model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
        optimizer = AdamW(model.parameters(), lr=args.lr_ft, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))

    for ep in range(args.warmup_epochs, args.epochs):
        tr_l, tr_a = run_epoch(True)
        va_l, va_a = run_epoch(False)
        scheduler.step()

        if va_a > best_acc:
            best_acc = va_a
            torch.save(model.state_dict(), best_path)
        print(f"[{args.model}] Epoch {ep+1}/{args.epochs} "
              f"| train acc={tr_a:.4f} val acc={va_a:.4f} best={best_acc:.4f}")

    print(f"Training finished. Best val acc: {best_acc:.4f}. Saved model to {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name (folder under src/models)")
    ap.add_argument("--data_root", default="datasets")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4, help="LR during warm-up or scratch training")
    ap.add_argument("--lr_ft", type=float, default=2e-4, help="LR when fine-tuning backbone")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--train_per_class", type=int, default=None,
                    help="Use all data if None; else limit per class")
    ap.add_argument("--val_per_class", type=int, default=None)
    ap.add_argument("--aug_strength", default="standard",
                    choices=["none", "standard", "plus"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()

    train_once(args)
