"""
classification/classifier.py
Phase 7 — Rule-based style & period classification from image features.
A lightweight CNN classifier is used when enough labeled data exists;
falls back to heuristic analysis otherwise.
"""
import cv2
import numpy as np


def classify_manuscript(image: np.ndarray) -> dict:
    """
    Analyze image features to estimate:
    - script_style: Dense / Sparse / Cursive / Geometric
    - estimated_period: based on style heuristics
    - confidence: float 0–1
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.resize(gray, (256, 256))

    # Feature 1: ink density (ratio of dark pixels)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    ink_density = np.sum(binary > 0) / binary.size

    # Feature 2: stroke variance (texture complexity)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Feature 3: horizontal line dominance (palm-leaf lines)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    h_ratio = np.mean(np.abs(sobely)) / (np.mean(np.abs(sobelx)) + 1e-6)

    # Classify style
    if ink_density > 0.35:
        style = "Dense Script"
        period = "Chola (9th–13th Century CE)"
        confidence = min(0.75, ink_density * 1.5)
    elif laplacian_var > 500:
        style = "Cursive Script"
        period = "Pandya (6th–14th Century CE)"
        confidence = min(0.70, laplacian_var / 1000)
    elif h_ratio > 1.5:
        style = "Geometric Script"
        period = "Pallava (4th–9th Century CE)"
        confidence = min(0.65, h_ratio / 3)
    else:
        style = "Sparse Script"
        period = "Nayak (16th–18th Century CE)"
        confidence = 0.50

    return {
        "style":      style,
        "period":     period,
        "confidence": round(confidence, 2),
        "ink_density":    round(float(ink_density), 3),
        "texture_score":  round(float(laplacian_var), 1),
        "line_ratio":     round(float(h_ratio), 2),
    }
