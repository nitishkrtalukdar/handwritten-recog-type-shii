"""
preprocess.py
-------------
Cleans up a photographed/scanned invoice image before field extraction:
  1. Orientation fix (EXIF)
  2. Grayscale + denoise
  3. Deskew (corrects small rotation from crooked phone photos)
  4. Adaptive contrast so faint pencil/pen strokes are easier for OCR to pick up

The output is a normalized RGB image resized to match the template's
reference resolution (so the ROI boxes calibrated on one sample line up
correctly on every new photo of the same physical invoice layout).
"""

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image_fix_orientation(path: str) -> np.ndarray:
    """Load with PIL first so EXIF rotation (common on phone photos) is respected."""
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)  # auto-rotate per EXIF tag
    pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Estimate and correct small rotation using the minAreaRect of all ink pixels.
    Works well for invoices since the page/table edges dominate the ink mask.
    Only corrects small angles (<15 deg) to avoid over-rotating on noisy scans.
    """
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > 15 or abs(angle) < 0.1:
        return gray  # skip: either noise or already straight
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def resize_to_reference(img: np.ndarray, ref_w: int, ref_h: int) -> np.ndarray:
    """Resize (not crop) so normalized ROI fractions from the template map correctly."""
    return cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC)


def denoise_and_enhance(bgr: np.ndarray) -> np.ndarray:
    """Light denoise + CLAHE contrast boost, kept in color (TrOCR expects RGB)."""
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def preprocess_invoice(path: str, ref_w: int, ref_h: int) -> np.ndarray:
    """
    Full pipeline: load -> deskew -> resize to template's reference size -> enhance.
    Returns a BGR numpy array ready for ROI cropping.
    """
    bgr = load_image_fix_orientation(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_deskewed = deskew(gray)

    # Re-apply the same rotation to the color image so color crops stay aligned.
    if gray_deskewed.shape == gray.shape and not np.array_equal(gray_deskewed, gray):
        h, w = gray.shape
        inv = cv2.bitwise_not(gray)
        _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(bw > 0))
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if 0.1 < abs(angle) <= 15:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            bgr = cv2.warpAffine(
                bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

    resized = resize_to_reference(bgr, ref_w, ref_h)
    enhanced = denoise_and_enhance(resized)
    return enhanced
