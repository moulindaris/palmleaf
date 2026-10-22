"""
training/datasets/tamil_dataset.py
Loaders tailored to the exact dataset structure:
  FINAL DATASET/DATASET/<1..71>/Letter<N>_<i>.png

  - 71 Tamil character classes
  - ~100 images per class
  - 7157 images total

Provides:
  1. TamilCharDataset     — for character recognition (CNN)
  2. TamilRestoreDataset  — for image restoration (U-Net)
"""

import os
import io
import zipfile
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMG_SIZE   = 64    # character images are small — 64x64 is enough
REST_SIZE  = 256   # restoration model uses 256x256
NUM_CLASSES = 71   # folders 1..71

ZIP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "archive (9).zip"
)


def _load_zip_entries():
    """
    Returns dict: { class_idx(0-based): [zip_entry_path, ...] }
    Reads directly from ZIP — no extraction needed.
    """
    z       = zipfile.ZipFile(ZIP_PATH)
    entries = {}
    for name in z.namelist():
        parts = name.replace("\\", "/").split("/")
        # Structure: FINAL DATASET/DATASET/<folder>/<file>.png
        if len(parts) == 4 and parts[3].lower().endswith(".png"):
            folder = parts[2]
            if folder.isdigit():
                idx = int(folder) - 1   # 0-based
                entries.setdefault(idx, []).append(name)
    z.close()
    return entries


def _read_image_from_zip(z: zipfile.ZipFile, path: str,
                          size: int, grayscale: bool = True) -> np.ndarray:
    data = z.read(path)
    img  = Image.open(io.BytesIO(data)).convert("L" if grayscale else "RGB")
    img  = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


# ---------------------------------------------------------------------------
# Damage augmentation
# ---------------------------------------------------------------------------

def add_damage(img: np.ndarray) -> np.ndarray:
    d = img.copy()
    h, w = d.shape
    for _ in range(np.random.randint(2, 8)):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        cv2.line(d, (x1, y1), (x2, y2), 0, np.random.randint(1, 3))
    for _ in range(np.random.randint(1, 4)):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(d, (cx, cy), np.random.randint(3, 12), 0, -1)
    noise = np.random.rand(h, w)
    d[noise < 0.04] = 0
    d[noise > 0.96] = 255
    if np.random.rand() > 0.5:
        d = cv2.GaussianBlur(d, (3, 3), 0)
    return d


# ---------------------------------------------------------------------------
# 1. Tamil Character Recognition Dataset
# ---------------------------------------------------------------------------

class TamilCharDataset(Dataset):
    """
    71-class Tamil character recognition.
    Loads images directly from ZIP — no disk extraction needed.
    """
    def __init__(self, split: str = "train", val_ratio: float = 0.15,
                 img_size: int = IMG_SIZE):
        self.img_size = img_size
        self.z        = zipfile.ZipFile(ZIP_PATH)
        entries       = _load_zip_entries()

        all_samples = []
        for cls_idx, paths in entries.items():
            for p in paths:
                all_samples.append((p, cls_idx))

        np.random.seed(42)
        idx      = np.random.permutation(len(all_samples))
        split_at = int(len(idx) * (1 - val_ratio))
        chosen   = idx[:split_at] if split == "train" else idx[split_at:]
        self.samples = [all_samples[i] for i in chosen]

        print(f"[TamilCharDataset] {split}: {len(self.samples)} samples | "
              f"{NUM_CLASSES} classes")

        if split == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                        scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = _read_image_from_zip(self.z, path, self.img_size, grayscale=True)
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)

    def __del__(self):
        try:
            self.z.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 2. Restoration Dataset (uses same images as clean targets)
# ---------------------------------------------------------------------------

class TamilRestoreDataset(Dataset):
    """
    Uses all 7157 Tamil character images as clean targets.
    Generates damaged versions on-the-fly.
    Images resized to REST_SIZE x REST_SIZE for U-Net.
    """
    def __init__(self, img_size: int = REST_SIZE, repeat: int = 3):
        self.img_size  = img_size
        self.repeat    = repeat
        self.z         = zipfile.ZipFile(ZIP_PATH)
        self.to_tensor = transforms.ToTensor()

        entries = _load_zip_entries()
        self.all_paths = []
        for paths in entries.values():
            self.all_paths.extend(paths)

        print(f"[TamilRestoreDataset] {len(self.all_paths)} images × "
              f"{repeat} augmentations = {len(self)} samples")

    def __len__(self):
        return len(self.all_paths) * self.repeat

    def __getitem__(self, idx):
        path  = self.all_paths[idx % len(self.all_paths)]
        clean = _read_image_from_zip(self.z, path, self.img_size, grayscale=True)

        # Paste onto manuscript-like background for realism
        bg = np.ones((self.img_size, self.img_size), dtype=np.uint8) * 220
        offset = (self.img_size - clean.shape[0]) // 2
        bg[offset:offset+clean.shape[0], offset:offset+clean.shape[1]] = clean
        clean = bg

        damaged = add_damage(clean)
        return self.to_tensor(damaged), self.to_tensor(clean)

    def __del__(self):
        try:
            self.z.close()
        except Exception:
            pass
