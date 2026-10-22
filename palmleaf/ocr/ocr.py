"""
ocr.py
Extracts Tamil text from a restored image using Tesseract OCR.
Requires: tesseract installed with Tamil language pack (tam).
"""
import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# Uncomment and set path if Tesseract is not on PATH (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _prepare_ocr_image(image: np.ndarray) -> np.ndarray:
    """Normalize scan scale and contrast before asking Tesseract to read it."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    if h < 300 or w < 300:
        scale = max(300 / h, 300 / w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    # Preserve faint incisions while raising local contrast.
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _candidate_quality(text: str, confidence_data: dict) -> float:
    confidences = []
    for value in confidence_data.get("conf", []):
        try:
            value = float(value)
            if value >= 0:
                confidences.append(value)
        except (TypeError, ValueError):
            continue
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    tamil_chars = sum("\u0B80" <= char <= "\u0BFF" for char in text)
    tamil_ratio = tamil_chars / max(len(text.strip()), 1)
    return mean_confidence + min(tamil_ratio, 1.0) * 8 + min(len(text.split()), 40) * 0.1


def extract_text(image: np.ndarray, retry: bool = False) -> str:
    """
    Extract Tamil text from a grayscale or binary image.
    Returns extracted string or a fallback message.
    """
    if not TESSERACT_AVAILABLE:
        return "[pytesseract not installed. Run: pip install pytesseract]"

    binary = _prepare_ocr_image(image)

    try:
        page_modes = (6, 11, 4) if retry else (6,)
        candidates = []
        for page_mode in page_modes:
            config = f"--oem 3 --psm {page_mode}"
            text = pytesseract.image_to_string(binary, lang="tam", config=config).strip()
            if text:
                data = pytesseract.image_to_data(binary, lang="tam", config=config,
                                                 output_type=pytesseract.Output.DICT)
                candidates.append((_candidate_quality(text, data), text))
        return max(candidates, key=lambda item: item[0])[1] if candidates else "[No Tamil text detected]"
    except pytesseract.TesseractNotFoundError:
        return "[Tesseract not found. Install from https://github.com/UB-Mannheim/tesseract/wiki]"
    except Exception as e:
        return f"[OCR error: {str(e)}]"
