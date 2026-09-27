"""
ocr_engine.py
-------------
Two OCR backends, used deliberately for different content on the invoice:

  * TrOCR (microsoft/trocr-base-handwritten)
        A transformer encoder-decoder fine-tuned specifically on handwritten
        text (IAM dataset lineage). Used for every field the vendor/customer
        actually writes by hand: names, addresses, dates, item descriptions,
        amounts, GSTINs written in the blanks, etc.

        NOTE: model weights download from huggingface.co the first time you
        run this. That requires normal internet access on your machine.

  * Tesseract (pytesseract)
        A traditional OCR engine, very accurate on clean MACHINE-PRINTED
        text but poor on cursive handwriting. Used only for the parts of
        the invoice that are pre-printed on the stationery (in this project
        those are captured once as `static_fields` in the template instead,
        but this wrapper is kept so it can OCR printed regions on a NEW,
        not-yet-catalogued vendor template).

We tested both empirically against the real sample in this project:
Tesseract read the printed GSTIN/address/bank block correctly, but produced
garbage on every handwritten number/word -- confirming the split is
necessary, not just theoretical.
"""

from dataclasses import dataclass
import os
import shutil
import sys
from typing import Optional

# Prevent transformers from importing broken or incompatible system tensorflow packages
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import numpy as np
import pytesseract
from PIL import Image


@dataclass
class OcrResult:
    text: str
    confidence: float  # 0-100, best-effort. TrOCR has no native confidence
    # score, so we derive one from generation log-probs; Tesseract gives one
    # natively.


class TrOCREngine:
    """
    Lazily-loaded wrapper around HuggingFace's TrOCR. Loading is deferred to
    first use so importing this module (and running everything else in the
    pipeline) doesn't require torch/transformers or a model download unless
    handwriting OCR is actually invoked.
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", device: Optional[str] = None):
        self.model_name = model_name
        self._device = device
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        except OSError as e:
            raise RuntimeError(
                f"Could not load TrOCR model '{self.model_name}'. This downloads from "
                "huggingface.co the first time it runs, so this usually means either "
                "(a) no internet access right now, or (b) a firewall/proxy blocking "
                "huggingface.co. Once downloaded it's cached locally (~/.cache/huggingface) "
                f"and works offline after that.\nOriginal error: {e}"
            ) from e
        self._model.to(self._device)
        self._model.eval()

    def read(self, image: np.ndarray) -> OcrResult:
        """
        image: a BGR or RGB numpy crop containing ONE line (or short block)
        of handwritten text. TrOCR is a single-line recognizer, which is why
        the pipeline segments the item table into per-row crops first.
        """
        if image is None or image.size == 0:
            return OcrResult(text="", confidence=0.0)
        self._ensure_loaded()
        import torch

        pil_img = Image.fromarray(image).convert("RGB")
        pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_length=64,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

        # Derive a rough confidence from the average per-token softmax
        # probability of the chosen tokens (not a calibrated metric, but
        # useful as a relative "needs review" signal).
        confidence = 90.0
        try:
            scores = outputs.scores  # tuple of (batch, vocab) logits per step
            if scores:
                probs = []
                seq = outputs.sequences[0][1:]  # drop the BOS token
                for step_logits, token_id in zip(scores, seq):
                    p = torch.softmax(step_logits[0], dim=-1)[token_id].item()
                    probs.append(p)
                if probs:
                    confidence = round(100.0 * sum(probs) / len(probs), 1)
        except Exception:
            pass

        return OcrResult(text=text, confidence=confidence)


class TesseractEngine:
    """Wrapper around pytesseract for machine-printed regions."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd
        self._configured = False

    def _ensure_tesseract_configured(self):
        if self._configured:
            return
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            self._configured = True
            return
        if shutil.which(pytesseract.pytesseract.tesseract_cmd or "tesseract"):
            self._configured = True
            return
        candidate_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for p in candidate_paths:
            if os.path.isfile(p):
                pytesseract.pytesseract.tesseract_cmd = p
                self._configured = True
                return
        self._configured = True

    def read(self, image: np.ndarray, psm: int = 6) -> OcrResult:
        if image is None or image.size == 0:
            return OcrResult(text="", confidence=0.0)
        self._ensure_tesseract_configured()
        try:
            pil_img = Image.fromarray(image).convert("RGB")
            data = pytesseract.image_to_data(
                pil_img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError(
                "Tesseract OCR is not installed or not in PATH. Please install Tesseract OCR "
                "(e.g., https://github.com/UB-Mannheim/tesseract/wiki) or configure its path."
            ) from e
        words, confs = [], []
        for w, c in zip(data["text"], data["conf"]):
            w = w.strip()
            if w:
                words.append(w)
                try:
                    c = float(c)
                    if c >= 0:
                        confs.append(c)
                except ValueError:
                    pass
        text = " ".join(words).strip()
        confidence = round(sum(confs) / len(confs), 1) if confs else 0.0
        return OcrResult(text=text, confidence=confidence)
