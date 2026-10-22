"""
training/inference.py
Loads all trained models and exposes simple inference functions
used by the Flask app.
"""
import os
import sys
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import cv2

from models.restoration_model import get_model as get_unet, load_weights as load_unet_weights
from training.classification_model import get_classifier
from training.ocr_correction_model import get_seq2seq
from training.datasets.dataset_loader import CATEGORIES, IDX2CAT, IMG_SIZE

CKPT_DIR = os.path.join(ROOT, "models")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model registry — loaded once
# ---------------------------------------------------------------------------

_restorer   = None
_classifier = None
_corrector  = None
_vocab      = None


def _load_restorer():
    global _restorer
    if _restorer is not None:
        return _restorer
    model = get_unet(DEVICE)
    path  = os.path.join(CKPT_DIR, "restorer.pth")
    if os.path.exists(path):
        load_unet_weights(model, path, DEVICE)
        print(f"[Inference] Restorer loaded from {path}")
    else:
        print("[Inference] WARNING: restorer.pth not found — using untrained model.")
    model.eval()
    _restorer = model
    return _restorer


def _load_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    model = get_classifier(pretrained=False).to(DEVICE)
    path  = os.path.join(CKPT_DIR, "classifier.pth")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"[Inference] Classifier loaded from {path}")
        model.eval()
        _classifier = model
    else:
        print("[Inference] classifier.pth not found — skipping ML classification.")
        _classifier = None
    return _classifier


def _load_corrector():
    global _corrector, _vocab
    if _corrector is not None:
        return _corrector, _vocab

    vocab_path = os.path.join(CKPT_DIR, "ocr_vocab.pkl")
    model_path = os.path.join(CKPT_DIR, "ocr_corrector.pth")

    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        print("[Inference] ocr_corrector.pth / ocr_vocab.pkl not found — skipping ML OCR correction.")
        return None, None

    with open(vocab_path, "rb") as f:
        _vocab = pickle.load(f)

    model = get_seq2seq(_vocab["vocab_size"], DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    _corrector = model
    print(f"[Inference] OCR corrector loaded from {model_path}")
    return _corrector, _vocab


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------

def restore_image_ml(image: np.ndarray) -> np.ndarray:
    """
    Run the U-Net on overlapping native-resolution grayscale tiles.

    The network was trained on 256x256 patches. Resizing an entire page to a
    256x256 square destroys the fine strokes this model is meant to preserve,
    so tile the page instead and blend overlapping predictions. Keep most of
    the source pixels in the result because the learned restorer can be
    imperfect on manuscript styles outside its training set.
    """
    model   = _load_restorer()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    height, width = gray.shape[:2]
    tile_size = IMG_SIZE
    overlap = 64
    stride = tile_size - overlap

    def tile_starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size + 1, stride))
        edge = length - tile_size
        if starts[-1] != edge:
            starts.append(edge)
        return starts

    ys, xs = tile_starts(height), tile_starts(width)
    padded_h = max(height, tile_size)
    padded_w = max(width, tile_size)
    pad_h, pad_w = padded_h - height, padded_w - width
    padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    # Feather tile edges so seams from independent predictions do not show.
    axis = np.hanning(tile_size).astype(np.float32)
    weight = np.maximum(np.outer(axis, axis), 1e-3)
    accumulated = np.zeros((padded_h, padded_w), dtype=np.float32)
    weights = np.zeros_like(accumulated)

    locations = [(y, x) for y in ys for x in xs]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(locations), 8):
            batch_locations = locations[start:start + 8]
            batch = np.stack([
                padded[y:y + tile_size, x:x + tile_size]
                for y, x in batch_locations
            ]).astype(np.float32) / 255.0
            tensor = torch.from_numpy(batch).unsqueeze(1).to(DEVICE)
            predictions = model(tensor).squeeze(1).cpu().numpy() * 255.0
            for (y, x), prediction in zip(batch_locations, predictions):
                accumulated[y:y + tile_size, x:x + tile_size] += prediction * weight
                weights[y:y + tile_size, x:x + tile_size] += weight

    predicted = accumulated / np.maximum(weights, 1e-6)
    # A restrained residual blend protects original strokes and page contrast
    # from over-smoothing or tone shifts in a model trained on limited data.
    blended = gray.astype(np.float32) * 0.90 + predicted[:height, :width] * 0.10
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def classify_image_ml(image: np.ndarray) -> dict:
    """
    Run ResNet-18 classifier on an image.
    Returns { category, confidence, all_scores }.
    Falls back to heuristic classifier if model not trained.
    """
    from classification.classifier import classify_manuscript as heuristic

    model = _load_classifier()
    if model is None:
        return heuristic(image)

    import torchvision.transforms as T
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    top_idx  = int(np.argmax(probs))
    category = IDX2CAT[top_idx]
    conf     = float(probs[top_idx])

    all_scores = {CATEGORIES[i]: round(float(probs[i]), 3) for i in range(len(CATEGORIES))}

    return {
        "category":   category,
        "confidence": round(conf, 3),
        "all_scores": all_scores,
        "source":     "ml_classifier",
    }


def correct_text_ml(noisy_text: str) -> str:
    """
    Run Seq2Seq corrector on noisy OCR text.
    Falls back to rule-based cleaner if model not trained.
    """
    from utils.text_cleaner import clean_tamil_text

    model, vocab = _load_corrector()
    if model is None or vocab is None:
        return clean_tamil_text(noisy_text)

    if not noisy_text or noisy_text.startswith("["):
        return noisy_text

    char2idx = vocab["char2idx"]
    idx2char = vocab["idx2char"]
    unk_idx  = char2idx.get("<UNK>", 0)
    sos_idx  = char2idx["<SOS>"]
    eos_idx  = char2idx["<EOS>"]

    # Process line by line
    corrected_lines = []
    for line in noisy_text.splitlines():
        line = line.strip()
        if not line:
            continue
        encoded = [char2idx.get(ch, unk_idx) for ch in line[:128]]
        src = torch.tensor(encoded, dtype=torch.long).to(DEVICE)

        pred_ids = model.greedy_decode(src, sos_idx, eos_idx, max_len=128)
        pred_text = "".join(idx2char.get(i, "") for i in pred_ids
                            if i not in {char2idx["<EOS>"], char2idx["<PAD>"]})
        corrected_lines.append(pred_text if pred_text.strip() else line)

    result = "\n".join(corrected_lines)
    return result if result.strip() else "[No readable Tamil text found]"


def get_model_status() -> dict:
    """Returns which models are trained and available."""
    from utils.tamil_modernizer import model_status as get_tamil_model_status

    return {
        "restorer":   os.path.exists(os.path.join(CKPT_DIR, "restorer.pth")),
        "classifier": os.path.exists(os.path.join(CKPT_DIR, "classifier.pth")),
        "ocr_corrector": os.path.exists(os.path.join(CKPT_DIR, "ocr_corrector.pth")),
        "ocr_vocab":  os.path.exists(os.path.join(CKPT_DIR, "ocr_vocab.pkl")),
        "modernizer": get_tamil_model_status(),
    }
