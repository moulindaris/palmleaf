"""
train.py
Trains the UNetRestorer on synthetically damaged images.
Usage: python models/train.py
Saves trained weights to models/restorer.pth
"""
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.restoration_model import get_model

IMG_SIZE = 256
EPOCHS = 30
BATCH_SIZE = 4
LR = 1e-3
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "restorer.pth")


def add_damage(img: np.ndarray) -> np.ndarray:
    """Synthetically damage a grayscale image: scratches, blotches, noise."""
    damaged = img.copy()
    h, w = damaged.shape

    # Random black scratches
    for _ in range(np.random.randint(5, 15)):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        cv2.line(damaged, (x1, y1), (x2, y2), 0, np.random.randint(1, 4))

    # Random blotches
    for _ in range(np.random.randint(3, 8)):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        r = np.random.randint(5, 20)
        cv2.circle(damaged, (cx, cy), r, 0, -1)

    # Salt-and-pepper noise
    noise_mask = np.random.rand(h, w)
    damaged[noise_mask < 0.05] = 0
    damaged[noise_mask > 0.95] = 255

    return damaged


class ManuscriptDataset(Dataset):
    """
    Loads grayscale images from data/ as clean targets.
    Generates damaged versions on-the-fly as inputs.
    Falls back to synthetic white images with Tamil-like strokes if data/ is empty.
    """
    def __init__(self, size=500):
        self.images = []
        if os.path.isdir(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    path = os.path.join(DATA_DIR, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        self.images.append(img)

        self.synthetic = len(self.images) == 0
        self.size = size if self.synthetic else len(self.images) * 10

    def _make_synthetic(self) -> np.ndarray:
        """Generate a synthetic manuscript-like image."""
        img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 220
        for _ in range(np.random.randint(8, 20)):
            x1 = np.random.randint(10, IMG_SIZE - 10)
            y = np.random.randint(10, IMG_SIZE - 10)
            length = np.random.randint(20, 80)
            cv2.line(img, (x1, y), (x1 + length, y), 30, np.random.randint(1, 3))
        for _ in range(np.random.randint(5, 15)):
            cx = np.random.randint(10, IMG_SIZE - 10)
            cy = np.random.randint(10, IMG_SIZE - 10)
            cv2.circle(img, (cx, cy), np.random.randint(2, 6), 30, -1)
        return img

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.synthetic:
            clean = self._make_synthetic()
        else:
            clean = self.images[idx % len(self.images)].copy()

        damaged = add_damage(clean)

        to_tensor = transforms.ToTensor()
        return to_tensor(damaged), to_tensor(clean)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    dataset = ManuscriptDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = get_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"Dataset size: {len(dataset)} | Epochs: {EPOCHS}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for damaged, clean in loader:
            damaged, clean = damaged.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(damaged)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{EPOCHS}] Loss: {avg:.6f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()
