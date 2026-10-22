"""
training/train_tamil.py

Trains 2 models on the Tamil character dataset (archive (9).zip):
  1. U-Net Restorer        → models/restorer.pth
  2. Tamil Char CNN (71)   → models/tamil_char_model.pth
                           → models/tamil_char_classes.pkl

Usage:
  python training/train_tamil.py                   # train both
  python training/train_tamil.py --model restore
  python training/train_tamil.py --model classify
"""

import os
import sys
import argparse
import pickle
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.restoration_model import get_model as get_unet
from training.tamil_char_model import get_tamil_char_model, NUM_CLASSES
from training.datasets.tamil_dataset import TamilRestoreDataset, TamilCharDataset

CKPT_DIR = os.path.join(ROOT, "models")
LOG_DIR  = os.path.join(ROOT, "training", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, name):
        self.path = os.path.join(LOG_DIR, f"{name}.log")
        self.f    = open(self.path, "w", encoding="utf-8")

    def log(self, msg):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# 1. Train U-Net Restorer
# ---------------------------------------------------------------------------

def train_restoration(epochs: int = 30, batch_size: int = 16, lr: float = 1e-3):
    log = Logger("restoration")
    log.log("=" * 55)
    log.log("TRAINING: U-Net Image Restorer")
    log.log(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Device: {DEVICE}")
    log.log("=" * 55)

    dataset = TamilRestoreDataset(repeat=3)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=(DEVICE == "cuda"))

    model     = get_unet(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    start     = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for damaged, clean in loader:
            damaged, clean = damaged.to(DEVICE), clean.to(DEVICE)
            optimizer.zero_grad()
            out  = model(damaged)
            loss = criterion(out, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        scheduler.step()
        avg = total / len(loader)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "restorer.pth"))

        log.log(f"Epoch [{epoch:02d}/{epochs}] Loss: {avg:.6f} "
                f"Best: {best_loss:.6f} | {time.time()-start:.0f}s")

    log.log(f"\nDone. Best loss: {best_loss:.6f} | Saved: models/restorer.pth")
    log.close()


# ---------------------------------------------------------------------------
# 2. Train Tamil Character CNN
# ---------------------------------------------------------------------------

def train_classifier(epochs: int = 30, batch_size: int = 64, lr: float = 1e-3):
    log = Logger("tamil_char_classifier")
    log.log("=" * 55)
    log.log("TRAINING: Tamil Character CNN (71 classes)")
    log.log(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Device: {DEVICE}")
    log.log("=" * 55)

    train_ds = TamilCharDataset(split="train")
    val_ds   = TamilCharDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))

    model     = get_tamil_char_model(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    start    = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds    = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc = correct / total if total > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, "tamil_char_model.pth"))

        log.log(f"Epoch [{epoch:02d}/{epochs}] "
                f"Loss: {train_loss/len(train_loader):.4f} "
                f"Val Acc: {acc*100:.1f}% "
                f"Best: {best_acc*100:.1f}% | {time.time()-start:.0f}s")

    # Save class mapping
    class_map = {i: str(i + 1) for i in range(NUM_CLASSES)}
    with open(os.path.join(CKPT_DIR, "tamil_char_classes.pkl"), "wb") as f:
        pickle.dump(class_map, f)

    log.log(f"\nDone. Best val accuracy: {best_acc*100:.1f}%")
    log.log(f"Saved: models/tamil_char_model.pth")
    log.log(f"Saved: models/tamil_char_classes.pkl")
    log.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=["restore", "classify", "all"],
                        default="all")
    parser.add_argument("--epochs_restore",   type=int, default=30)
    parser.add_argument("--epochs_classify",  type=int, default=30)
    parser.add_argument("--batch_size",       type=int, default=32)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("Tamil Palm-Leaf — Character Recognition Training")
    print(f"Dataset: 7157 images | 71 Tamil character classes")
    print(f"Device : {DEVICE}")
    print(f"{'='*55}\n")

    if args.model in ("restore", "all"):
        train_restoration(
            epochs=args.epochs_restore,
            batch_size=args.batch_size
        )

    if args.model in ("classify", "all"):
        train_classifier(
            epochs=args.epochs_classify,
            batch_size=args.batch_size * 2
        )

    print("\nAll training complete!")
    print("Files saved in models/:")
    print("  restorer.pth")
    print("  tamil_char_model.pth")
    print("  tamil_char_classes.pkl")
