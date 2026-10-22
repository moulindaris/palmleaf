"""
training/train_all.py

Trains all 3 models sequentially:
  1. U-Net Image Restorer       → models/restorer.pth
  2. ResNet-18 Classifier       → models/classifier.pth
  3. Seq2Seq OCR Corrector      → models/ocr_corrector.pth + models/ocr_vocab.pkl

Usage:
  python training/train_all.py                    # train all
  python training/train_all.py --model restoration
  python training/train_all.py --model classifier
  python training/train_all.py --model ocr
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
from training.classification_model import get_classifier
from training.ocr_correction_model import get_seq2seq
from training.datasets.dataset_loader import (
    RestorationDataset, ClassificationDataset, OCRCorrectionDataset
)

CKPT_DIR = os.path.join(ROOT, "models")
LOG_DIR  = os.path.join(ROOT, "training", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, name: str):
        self.path = os.path.join(LOG_DIR, f"{name}.log")
        self.f    = open(self.path, "w", encoding="utf-8")

    def log(self, msg: str):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# 1. Train Restoration (U-Net)
# ---------------------------------------------------------------------------

def train_restoration(epochs: int = 40, batch_size: int = 8, lr: float = 1e-3):
    log = Logger("restoration")
    log.log("=" * 50)
    log.log("TRAINING: U-Net Image Restorer")
    log.log("=" * 50)

    dataset = RestorationDataset(augment=True)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=(DEVICE == "cuda"))

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

        if epoch % 5 == 0 or epoch == 1:
            log.log(f"Epoch [{epoch:03d}/{epochs}] Loss: {avg:.6f} "
                    f"Best: {best_loss:.6f} | {time.time()-start:.1f}s")

    log.log(f"\nRestoration training done. Best loss: {best_loss:.6f}")
    log.log(f"Saved: models/restorer.pth")
    log.close()


# ---------------------------------------------------------------------------
# 2. Train Classifier (ResNet-18)
# ---------------------------------------------------------------------------

def train_classifier(epochs: int = 30, batch_size: int = 16, lr: float = 1e-4):
    log = Logger("classifier")
    log.log("=" * 50)
    log.log("TRAINING: ResNet-18 Category Classifier")
    log.log("=" * 50)

    try:
        train_ds = ClassificationDataset(split="train")
        val_ds   = ClassificationDataset(split="val")
    except (FileNotFoundError, ValueError) as e:
        log.log(f"[SKIP] {e}")
        log.close()
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = get_classifier(pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 10,
        steps_per_epoch=len(train_loader), epochs=epochs
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
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc = correct / total if total > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "classifier.pth"))

        if epoch % 5 == 0 or epoch == 1:
            log.log(f"Epoch [{epoch:03d}/{epochs}] "
                    f"Loss: {train_loss/len(train_loader):.4f} "
                    f"Val Acc: {acc*100:.1f}% Best: {best_acc*100:.1f}% "
                    f"| {time.time()-start:.1f}s")

    log.log(f"\nClassifier training done. Best val accuracy: {best_acc*100:.1f}%")
    log.log(f"Saved: models/classifier.pth")
    log.close()


# ---------------------------------------------------------------------------
# 3. Train OCR Corrector (Seq2Seq)
# ---------------------------------------------------------------------------

def train_ocr(epochs: int = 50, batch_size: int = 32, lr: float = 5e-4):
    log = Logger("ocr_corrector")
    log.log("=" * 50)
    log.log("TRAINING: Seq2Seq OCR Text Corrector")
    log.log("=" * 50)

    try:
        dataset = OCRCorrectionDataset(max_len=128)
    except FileNotFoundError as e:
        log.log(f"[SKIP] {e}")
        log.close()
        return

    # Save vocabulary for inference
    vocab_path = os.path.join(CKPT_DIR, "ocr_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({
            "char2idx": dataset.char2idx,
            "idx2char": dataset.idx2char,
            "vocab_size": dataset.vocab_size,
        }, f)
    log.log(f"Vocabulary saved: {vocab_path}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model     = get_seq2seq(dataset.vocab_size, DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    pad_idx   = dataset.char2idx["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_loss = float("inf")
    start     = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        # Reduce teacher forcing over time
        tf_ratio = max(0.3, 1.0 - epoch / epochs)

        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, tgt_in, teacher_forcing_ratio=tf_ratio)
            # output: (B, T, V), tgt_out: (B, T)
            B, T, V = output.shape
            loss = criterion(output[:, 1:].reshape(-1, V),
                             tgt_out[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        scheduler.step(avg)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "ocr_corrector.pth"))

        if epoch % 10 == 0 or epoch == 1:
            log.log(f"Epoch [{epoch:03d}/{epochs}] Loss: {avg:.4f} "
                    f"Best: {best_loss:.4f} TF: {tf_ratio:.2f} "
                    f"| {time.time()-start:.1f}s")

        # Sample prediction every 10 epochs
        if epoch % 10 == 0:
            _sample_prediction(model, dataset, log)

    log.log(f"\nOCR corrector training done. Best loss: {best_loss:.4f}")
    log.log(f"Saved: models/ocr_corrector.pth")
    log.close()


def _sample_prediction(model, dataset, log):
    """Show a sample correction during training."""
    model.eval()
    src, _, tgt_out = dataset[0]
    src = src.to(DEVICE)
    pred_ids = model.greedy_decode(
        src,
        sos_idx=dataset.char2idx["<SOS>"],
        eos_idx=dataset.char2idx["<EOS>"]
    )
    noisy  = dataset.pairs[0][0]
    clean  = dataset.pairs[0][1]
    pred   = dataset.decode(pred_ids)
    log.log(f"  Sample | Noisy:  {noisy[:60]}")
    log.log(f"         | Target: {clean[:60]}")
    log.log(f"         | Pred:   {pred[:60]}")
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["restoration", "classifier", "ocr", "all"],
                        default="all")
    parser.add_argument("--epochs_restoration", type=int, default=40)
    parser.add_argument("--epochs_classifier",  type=int, default=30)
    parser.add_argument("--epochs_ocr",         type=int, default=50)
    parser.add_argument("--batch_size",          type=int, default=8)
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Tamil Palm-Leaf Manuscript — Multi-Model Training")
    print(f"Device: {DEVICE}")
    print(f"{'='*50}\n")

    if args.model in ("restoration", "all"):
        train_restoration(epochs=args.epochs_restoration, batch_size=args.batch_size)

    if args.model in ("classifier", "all"):
        train_classifier(epochs=args.epochs_classifier, batch_size=args.batch_size * 2)

    if args.model in ("ocr", "all"):
        train_ocr(epochs=args.epochs_ocr, batch_size=args.batch_size * 4)

    print("\nAll training complete.")
