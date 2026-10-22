import cv2
import numpy as np

class ImagePreprocessor:
    @staticmethod
    def preprocess_palm_leaf(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        background = cv2.medianBlur(gray, 21)
        diff = cv2.absdiff(gray, background)
        normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return denoised
