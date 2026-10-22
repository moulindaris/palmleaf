"""
helpers.py
Utility functions: image I/O, base64 encoding, Tamil text correction.
"""
import re
import base64
import os
import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


def save_image(image: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def to_base64(image: np.ndarray, ext: str = ".png") -> str:
    """Encode a numpy image to a base64 data-URI string for HTML embedding."""
    _, buf = cv2.imencode(ext, image)
    b64 = base64.b64encode(buf).decode("utf-8")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Restoration inference
# ---------------------------------------------------------------------------

def restore_image(image: np.ndarray, model, device: str = "cpu") -> np.ndarray:
    """
    Run the UNet restorer on a grayscale image.
    Handles arbitrary sizes by resizing to 256x256 and back.
    """
    orig_h, orig_w = image.shape[:2]

    # Prepare input
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    resized = cv2.resize(gray, (256, 256))
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,256,256)

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    restored = output.squeeze().cpu().numpy()  # (256,256) float [0,1]
    restored = (restored * 255).clip(0, 255).astype(np.uint8)
    restored = cv2.resize(restored, (orig_w, orig_h))
    return restored


# ---------------------------------------------------------------------------
# Tamil text correction
# ---------------------------------------------------------------------------

# Characters that are clearly OCR noise (non-Tamil, non-punctuation)
_NOISE_PATTERN = re.compile(r"[^\u0B80-\u0BFF\s\.,;:\-\(\)\d]")

# Common OCR substitution fixes for Tamil
_SUBSTITUTIONS = [
    (re.compile(r"\|"), "।"),   # pipe often misread as danda
    (re.compile(r"0"), "௦"),    # digit zero → Tamil zero
    (re.compile(r"\s{2,}"), " "),  # collapse multiple spaces
]


def correct_tamil_text(text: str) -> str:
    """
    Light cleanup of Tesseract Tamil output:
    - Remove non-Tamil noise characters
    - Apply common substitutions
    - Strip blank lines
    """
    if not text or text.startswith("["):
        return text  # pass-through error messages

    # Remove noise characters
    text = _NOISE_PATTERN.sub("", text)

    # Apply substitutions
    for pattern, replacement in _SUBSTITUTIONS:
        text = pattern.sub(replacement, text)

    # Remove empty lines and strip
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines) if lines else "[No readable Tamil text found]"
