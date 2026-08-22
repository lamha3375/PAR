"""
Training Pipeline for UPAR_UNIFIED Dataset on HydraPlus-Net in D:\\AI DATASET.
Supports Masked BCE Loss for Unknown labels, mixed precision, checkpointing, and comprehensive metric evaluation.
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from datasets.upar.loader import UPARDataset, get_upar_transforms
from models.hydraplus.par_model import UnifiedPARModel
from training.loss import MaskedBCEWithLogitsLoss
from training.evaluate import evaluate_model

BAR_WIDTH = 30


class KerasProgressBar:
    def __init__(self, total: int, epoch: int, total_epochs: int):
        self.total = total
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.current = 0

    def _fmt_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"

    def _fmt_rate(self, elapsed: float) -> str:
        if self.current == 0:
            return "?ms/step"
        ms = elapsed / self.current * 1000
        if ms >= 1000:
            return f"{ms / 1000:.1f}s/step"
        return f"{int(ms)}ms/step"

    def _bar_str(self, filled: int) -> str:
        return '\u2501' * filled + ' ' * (BAR_WIDTH - filled)

    def _build_line(self, bar: str, elapsed: float, loss: float, acc: float, val_metrics: dict = None) -> str:
        parts = [
            f"\r{self.current}/{self.total} {bar}",
            self._fmt_time(elapsed),
            self._fmt_rate(elapsed),
            f"- accuracy: {acc:.4f}",
            f"- loss: {loss:.4f}",
        ]
        if val_metrics is not None:
            parts += [
                f"- val_accuracy: {val_metrics.get('accuracy', 0):.4f}",
                f"- val_loss: {val_metrics.get('loss', 0):.4f}",
                f"- val_mA: {val_metrics.get('mA', 0):.4f}",
                f"- val_f1: {val_metrics.get('f1', 0):.4f}",
            ]
        return ' '.join(parts)

    def update(self, loss: float, accuracy: float):
        self.current += 1
        elapsed = time.time() - self.start_time
        filled = int(BAR_WIDTH * self.current / self.total)
        line = self._build_line(self._bar_str(filled), elapsed, loss, accuracy)
        sys.stdout.write(line)
        sys.stdout.flush()

    def finalize(self, loss: float, accuracy: float, val_metrics: dict):
        elapsed = time.time() - self.start_time
        line = self._build_line(self._bar_str(BAR_WIDTH), elapsed, loss, accuracy, val_metrics)
        sys.stdout.write(line + '\n')
        sys.stdout.flush()


def train_upar(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}", flush=True)
    print(f"  UPAR_UNIFIED PAR Training  |  Device: {str(device).upper()}", flush=True)
    print(f"  Backbone: {args.backbone}  |  Batch: {args.batch_size}"
          f"  |  Epochs: {args.epochs}  |  LR: {args.lr}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    checkpoint_dir = os.path.join(args.unified_root, "checkpoints")
    reports_dir = os.path.join(args.unified_root, "reports")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    train_transform, val_transform = get_upar_transforms(
        height=args.img_height, width=args.img_width
    )

    print(" Loading Train Dataset...", flush=True)
    train_dataset = UPARDataset(
        root_dir=args.unified_root,
        split='train',
        transform=train_transform,
        sample_ratio=args.sample_ratio
    )

    print(" Loading Val Dataset...", flush=True)
    val_dataset = UPARDataset(
        root_dir=args.unified_root,
        split='val',
        transform=val_transform,
        sample_ratio=args.sample_ratio
    )

    print(" Loading Test Dataset...", flush=True)
    test_dataset = UPARDataset(
        root_dir=args.unified_root,
        split='test',
        transform=val_transform,
        sample_ratio=args.sample_ratio
    )

    print(f" Train: {len(train_dataset):,} samples | Val: {len(val_dataset):,} samples | Test: {len(test_dataset):,} samples\n", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = UnifiedPARModel(
        num_attributes=train_dataset.num_attributes,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout
    ).to(device)

    loss_builder = MaskedBCEWithLogitsLoss()
    pos_weights = loss_builder.compute_pos_weights(train_dataset.labels)
    criterion = MaskedBCEWithLogitsLoss(pos_weights=pos_weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_ma = 0.0
    best_epoch = 0
    best_model_path = os.path.join(checkpoint_dir, "hydraplus_upar_best.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)

        model.train()
        running_loss = 0.0
        running_acc = 0.0

        pbar = KerasProgressBar(
            total=len(train_loader), epoch=epoch, total_epochs=args.epochs
        )

        for step, (imgs, targets, _, _) in enumerate(train_loader, 1):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if hasattr(torch.amp, 'autocast'):
                autocast_ctx = torch.amp.autocast(
                    device_type=device.type, enabled=(device.type == 'cuda')
                )
            else:
                autocast_ctx = torch.cuda.amp.autocast(
                    enabled=(device.type == 'cuda')
                )

            with autocast_ctx:
                logits = model(imgs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                valid_mask = (targets != 2.0)
                batch_acc = (preds[valid_mask] == targets[valid_mask]).float().mean().item() if valid_mask.sum() > 0 else 1.0

            running_loss += loss.item()
            running_acc += batch_acc

            pbar.update(
                loss=running_loss / step,
                accuracy=running_acc / step
            )

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)

        val_metrics = evaluate_model(model, val_loader, device, criterion=criterion)
        val_ma = val_metrics["mA"]

        pbar.finalize(
            loss=epoch_loss,
            accuracy=epoch_acc,
            val_metrics=val_metrics
        )

        epoch_ckpt_path = os.path.join(checkpoint_dir, f"hydraplus_upar_epoch_{epoch:03d}.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_ma": val_ma,
            "val_metrics": val_metrics,
            "backbone": args.backbone,
            "num_attributes": train_dataset.num_attributes,
            "attr_names": train_dataset.attr_names
        }
        torch.save(checkpoint, epoch_ckpt_path)

        if val_ma > best_val_ma:
            best_val_ma = val_ma
            best_epoch = epoch
            torch.save(checkpoint, best_model_path)
            print(f"  --> Best Checkpoint saved: {best_model_path} (val_mA: {val_ma * 100:.2f}%)", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(" Running Final Evaluation on Test Set...", flush=True)
    print(f"{'=' * 70}\n", flush=True)
    
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        
    test_metrics = evaluate_model(model, test_loader, device, criterion=criterion)
    test_ma = test_metrics["mA"]
    
    print(f" Test mA: {test_ma * 100:.2f}% | Test F1: {test_metrics['f1'] * 100:.2f}% | Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%\n")

    training_report_txt = os.path.join(reports_dir, "training_report.txt")
    metrics_csv = os.path.join(reports_dir, "metrics.csv")
    per_attr_csv = os.path.join(reports_dir, "per_attribute_metrics.csv")

    with open(training_report_txt, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("HYDRAPLUS-NET UPAR_UNIFIED TRAINING REPORT\n")
        f.write("========================================\n\n")
        f.write(f"Dataset: UPAR_UNIFIED (Market1501 + PA100k + PETA)\n")
        f.write(f"Model: HydraPlus-Net / UnifiedPARModel\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Number of attributes: {train_dataset.num_attributes}\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Val samples: {len(val_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Epochs trained: {args.epochs}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation mA: {best_val_ma * 100:.2f}%\n")
        f.write(f"Test mA: {test_ma * 100:.2f}%\n")
        f.write(f"Test F1: {test_metrics['f1'] * 100:.2f}%\n")

    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("Metric,Value\n")
        f.write(f"Best_Epoch,{best_epoch}\n")
        f.write(f"Best_Val_mA,{best_val_ma:.4f}\n")
        f.write(f"Test_mA,{test_ma:.4f}\n")
        f.write(f"Test_Accuracy,{test_metrics['accuracy']:.4f}\n")
        f.write(f"Test_Precision,{test_metrics['precision']:.4f}\n")
        f.write(f"Test_Recall,{test_metrics['recall']:.4f}\n")
        f.write(f"Test_F1,{test_metrics['f1']:.4f}\n")

    with open(per_attr_csv, "w", encoding="utf-8") as f:
        f.write("Attribute_ID,Attribute_Name,mA,Precision,Recall,F1\n")
        for idx, attr in enumerate(train_dataset.attr_names):
            m = test_metrics["per_attribute"].get(attr, {})
            f.write(f"{idx:02d},{attr},{m.get('mA',0):.4f},{m.get('precision',0):.4f},{m.get('recall',0):.4f},{m.get('f1',0):.4f}\n")

    print(f"Saved reports:\n  - {training_report_txt}\n  - {metrics_csv}\n  - {per_attr_csv}")
    print("\nTRAINING COMPLETED SUCCESSFULLY.")
    return model, test_ma


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train HydraPlus-Net on UPAR_UNIFIED Dataset")
    parser.add_argument('--unified_root', type=str, default=r'D:\AI DATASET\UPAR_UNIFIED')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=128)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--num_workers', type=int, default=0)

    cli_args = parser.parse_args()
    train_upar(cli_args)
