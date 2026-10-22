"""
training/datasets/dataset_loader.py

Handles dataset loading for all 3 training tasks:
  1. Restoration  — pairs of (damaged_image, clean_image)
  2. Classification — (image, category_label)
  3. OCR Correction — pairs of (noisy_text, clean_text)

Expected folder structure in data/:
  data/
  ├── restoration/
  │   ├── clean/          ← original clean manuscript images
  │   └── damaged/        ← corresponding damaged versions (optional, auto-generated if missing)
  ├── classification/
  │   ├── Medicine/       ← images per category
  │   ├── Astrology/
  │   ├── Literature/
  │   ├── Philosophy/
  │   ├── Mathematics/
  │   └── History/
  └── ocr/
      ├── noisy.txt       ← one noisy OCR line per line
      └── clean.txt       ← corresponding clean Tamil text line per line
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")

IMG_SIZE = 256

CATEGORIES = ["Medicine", "Astrology", "Literature", "Philosophy", "Mathematics", "History"]
CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
IDX2CAT = {i: c for c, i in CAT2IDX.items()}


# ---------------------------------------------------------------------------
# Damage augmentation (shared)
# ---------------------------------------------------------------------------

def add_damage(img: np.ndarray) -> np.ndarray:
    damaged = img.copy()
    h, w = damaged.shape[:2]
    # Scratches
    for _ in range(np.random.randint(3, 12)):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        cv2.line(damaged, (x1, y1), (x2, y2), 0, np.random.randint(1, 4))
    # Blotches
    for _ in range(np.random.randint(2, 7)):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(damaged, (cx, cy), np.random.randint(5, 25), 0, -1)
    # Salt-and-pepper
    noise = np.random.rand(h, w)
    damaged[noise < 0.04] = 0
    damaged[noise > 0.96] = 255
    # Gaussian blur (simulate fading)
    if np.random.rand() > 0.5:
        damaged = cv2.GaussianBlur(damaged, (3, 3), 0)
    return damaged


def _load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load: {path}")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def _load_color(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


# ---------------------------------------------------------------------------
# 1. Restoration Dataset
# ---------------------------------------------------------------------------

class RestorationDataset(Dataset):
    """
    Loads clean images from data/restoration/clean/.
    Generates damaged versions on-the-fly.
    Falls back to synthetic data if folder is empty.
    """
    def __init__(self, augment: bool = True):
        self.augment   = augment
        self.to_tensor = transforms.ToTensor()
        self.images    = []

        clean_dir = os.path.join(DATA_DIR, "restoration", "clean")
        if os.path.isdir(clean_dir):
            for f in os.listdir(clean_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.images.append(os.path.join(clean_dir, f))

        self.synthetic = len(self.images) == 0
        if self.synthetic:
            print("[RestorationDataset] No images found — using synthetic data.")
        else:
            print(f"[RestorationDataset] Loaded {len(self.images)} clean images.")

    def _make_synthetic(self) -> np.ndarray:
        img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 220
        for _ in range(np.random.randint(8, 20)):
            x1 = np.random.randint(10, IMG_SIZE - 10)
            y  = np.random.randint(10, IMG_SIZE - 10)
            cv2.line(img, (x1, y), (x1 + np.random.randint(20, 80), y),
                     30, np.random.randint(1, 3))
        for _ in range(np.random.randint(5, 15)):
            cv2.circle(img,
                       (np.random.randint(10, IMG_SIZE-10), np.random.randint(10, IMG_SIZE-10)),
                       np.random.randint(2, 6), 30, -1)
        return img

    def __len__(self):
        return max(2000, len(self.images) * 20) if not self.synthetic else 2000

    def __getitem__(self, idx):
        if self.synthetic:
            clean = self._make_synthetic()
        else:
            clean = _load_gray(self.images[idx % len(self.images)])

        damaged = add_damage(clean)
        return self.to_tensor(damaged), self.to_tensor(clean)


# ---------------------------------------------------------------------------
# 2. Classification Dataset
# ---------------------------------------------------------------------------

class ClassificationDataset(Dataset):
    """
    Loads images from data/classification/<Category>/ folders.
    Applies augmentation for training.
    """
    def __init__(self, split: str = "train", val_ratio: float = 0.15):
        self.samples = []  # list of (path, label_idx)

        cls_dir = os.path.join(DATA_DIR, "classification")
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Classification data not found at {cls_dir}\n"
                f"Create folders: {cls_dir}/<Category>/ and place images inside."
            )

        all_samples = []
        for cat in CATEGORIES:
            cat_dir = os.path.join(cls_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            for f in os.listdir(cat_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    all_samples.append((os.path.join(cat_dir, f), CAT2IDX[cat]))

        if not all_samples:
            raise ValueError("No images found in classification folders.")

        # Deterministic split
        np.random.seed(42)
        idx = np.random.permutation(len(all_samples))
        split_at = int(len(idx) * (1 - val_ratio))
        chosen = idx[:split_at] if split == "train" else idx[split_at:]
        self.samples = [all_samples[i] for i in chosen]

        print(f"[ClassificationDataset] {split}: {len(self.samples)} samples "
              f"across {len(CATEGORIES)} categories.")

        # Transforms
        if split == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = _load_color(path)
        return self.transform(img), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# 3. OCR Correction Dataset
# ---------------------------------------------------------------------------

class OCRCorrectionDataset(Dataset):
    """
    Loads paired noisy/clean Tamil text lines from:
      data/ocr/noisy.txt
      data/ocr/clean.txt

    Builds character-level vocabulary automatically.
    """
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(self, max_len: int = 128):
        self.max_len = max_len
        noisy_path = os.path.join(DATA_DIR, "ocr", "noisy.txt")
        clean_path = os.path.join(DATA_DIR, "ocr", "clean.txt")

        if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
            raise FileNotFoundError(
                f"OCR data not found.\nExpected:\n  {noisy_path}\n  {clean_path}"
            )

        with open(noisy_path, encoding="utf-8") as f:
            noisy_lines = [l.strip() for l in f if l.strip()]
        with open(clean_path, encoding="utf-8") as f:
            clean_lines = [l.strip() for l in f if l.strip()]

        assert len(noisy_lines) == len(clean_lines), \
            "noisy.txt and clean.txt must have the same number of lines."

        self.pairs = list(zip(noisy_lines, clean_lines))
        print(f"[OCRCorrectionDataset] Loaded {len(self.pairs)} text pairs.")

        # Build vocabulary from all characters
        all_chars = set()
        for n, c in self.pairs:
            all_chars.update(n)
            all_chars.update(c)

        specials = [self.PAD, self.SOS, self.EOS, self.UNK]
        vocab    = specials + sorted(all_chars)
        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(vocab)
        print(f"[OCRCorrectionDataset] Vocabulary size: {self.vocab_size}")

    def encode(self, text: str) -> list:
        unk = self.char2idx[self.UNK]
        return [self.char2idx.get(ch, unk) for ch in text]

    def decode(self, indices: list) -> str:
        stop = {self.char2idx[self.EOS], self.char2idx[self.PAD]}
        return "".join(
            self.idx2char[i] for i in indices
            if i not in stop and i in self.idx2char
        )

    def _pad(self, seq: list, length: int) -> torch.Tensor:
        pad_idx = self.char2idx[self.PAD]
        seq = seq[:length]
        seq += [pad_idx] * (length - len(seq))
        return torch.tensor(seq, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        sos = self.char2idx[self.SOS]
        eos = self.char2idx[self.EOS]

        src = self._pad(self.encode(noisy), self.max_len)
        tgt_in  = self._pad([sos] + self.encode(clean), self.max_len)
        tgt_out = self._pad(self.encode(clean) + [eos],  self.max_len)
        return src, tgt_in, tgt_out
