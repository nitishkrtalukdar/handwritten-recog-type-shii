"""
extractor.py
------------
Orchestrates the full per-invoice pipeline:

    image file
      -> preprocess (deskew/denoise, resize to template's reference size)
      -> crop each ROI from template.json
      -> OCR each ROI with the engine specified for it (trocr / tesseract)
      -> parse/clean the raw OCR text according to the field's type
      -> segment + OCR the variable-length item table separately
      -> assemble everything into one flat dict matching excel_writer.COLUMNS

The OCR engines are injected (not hard-coded) so this module can be unit
tested with a fake/mock engine that returns canned text instead of running
the real (slow, model-download-requiring) TrOCR -- see tests/test_extractor.py.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import field_parser as fp
from .ocr_engine import OcrResult, TesseractEngine, TrOCREngine
from .preprocess import preprocess_invoice
from .row_segmentation import segment_rows

LOW_CONFIDENCE_THRESHOLD = 35.0  # below this, a field's row gets flagged for human review


@dataclass
class FieldExtraction:
    name: str
    raw_text: str
    value: object
    confidence: float


@dataclass
class ExtractionReport:
    record: Dict
    fields: List[FieldExtraction] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)


class InvoiceExtractor:
    def __init__(
        self,
        template_path: str,
        trocr_engine: Optional[TrOCREngine] = None,
        tesseract_engine: Optional[TesseractEngine] = None,
        verify_static_fields_with_ocr: bool = False,
    ):
        with open(template_path, "r") as f:
            self.template = json.load(f)
        self.ref_w = self.template["reference_image_size"]["width"]
        self.ref_h = self.template["reference_image_size"]["height"]
        self.static_fields = self.template.get("static_fields", {})
        self.verify_static_fields_with_ocr = verify_static_fields_with_ocr

        self.trocr = trocr_engine or TrOCREngine()
        self.tesseract = tesseract_engine or TesseractEngine()

    # ---------- low-level helpers ----------

    def _crop_norm_box(self, img: np.ndarray, box_norm) -> np.ndarray:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = box_norm
        px = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
        return img[px[1]: px[3], px[0]: px[2]]

    def _ocr_region(self, img: np.ndarray, region: dict) -> OcrResult:
        crop = self._crop_norm_box(img, region["box"])
        if crop is None or crop.size == 0:
            return OcrResult(text="", confidence=0.0)
        # OpenCV crops are BGR; convert to RGB for both engines' PIL conversion.
        rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
        if region["engine"] == "tesseract":
            return self.tesseract.read(rgb)
        return self.trocr.read(rgb)

    def _parse_by_type(self, field_type: str, raw: str):
        return {
            "text": fp.clean_text,
            "gstin": fp.clean_gstin,
            "date": fp.clean_date,
            "amount": fp.clean_amount,
            "percent": fp.clean_percent,
        }.get(field_type, fp.clean_text)(raw)

    # ---------- item table ----------

    def _extract_items_table(self, img: np.ndarray, region: dict):
        table = self._crop_norm_box(img, region["box"])
        cols = region["columns"]
        bands = segment_rows(table, tuple(cols["description"]))

        rows = []
        fields_out = []
        th, tw = table.shape[:2]
        for band in bands:
            row_img = table[band.y_start: band.y_end, :]
            row_data = {}
            for col_name, (fx0, fx1) in cols.items():
                x0, x1 = int(fx0 * tw), int(fx1 * tw)
                cell = row_img[:, x0:x1]
                if cell.size == 0:
                    row_data[col_name] = ""
                    continue
                rgb = cell[:, :, ::-1]
                result = self.trocr.read(rgb)
                fields_out.append(FieldExtraction(f"item.{col_name}", result.text, result.text, result.confidence))
                row_data[col_name] = result.text.strip()
            rows.append(row_data)

        descriptions = []
        hsn_codes = []
        taxable_total = 0.0
        any_taxable_parsed = False
        for i, r in enumerate(rows, start=1):
            desc = fp.clean_text(r.get("description", ""))
            rate = fp.clean_amount(r.get("rate", ""))
            taxable = fp.clean_amount(r.get("taxable_value", ""))
            hsn = fp.clean_text(r.get("hsn", ""))
            piece = desc
            if rate is not None:
                piece += f" (Rate: {rate})"
            descriptions.append(f"{i}) {piece}".strip())
            if hsn:
                hsn_codes.append(hsn)
            if taxable is not None:
                taxable_total += taxable
                any_taxable_parsed = True

        return {
            "descriptions_joined": "; ".join(descriptions),
            "hsn_joined": ", ".join(dict.fromkeys(hsn_codes)),  # de-dup, keep order
            "computed_taxable_total": taxable_total if any_taxable_parsed else None,
            "row_count": len(rows),
        }, fields_out

    # ---------- top-level ----------

    def extract(self, image_path: str) -> ExtractionReport:
        img = preprocess_invoice(image_path, self.ref_w, self.ref_h)

        fields_out: List[FieldExtraction] = []
        values: Dict[str, object] = {}

        simple_regions = {k: v for k, v in self.template["regions"].items() if v.get("type") != "table"}
        for name, region in simple_regions.items():
            result = self._ocr_region(img, region)
            parsed = self._parse_by_type(region["type"], result.text)
            fields_out.append(FieldExtraction(name, result.text, parsed, result.confidence))
            values[name] = parsed

        table_region = self.template["regions"].get("items_table")
        item_summary = {"descriptions_joined": "", "hsn_joined": "", "computed_taxable_total": None, "row_count": 0}
        if table_region:
            item_summary, item_fields = self._extract_items_table(img, table_region)
            fields_out.extend(item_fields)

        buyer_address = " ".join(
            part for part in [values.get("buyer_address_l1", ""), values.get("buyer_address_l2", "")] if part
        ).strip()

        static = self.static_fields
        record = {
            "Source File": os.path.basename(image_path),
            "Invoice No": values.get("invoice_no", ""),
            "Invoice Date": values.get("invoice_date", ""),
            "Supplier Name": static.get("supplier_name", ""),
            "Supplier Address": static.get("supplier_address", ""),
            "Supplier GSTIN": static.get("supplier_gstin", ""),
            "Buyer Name": values.get("buyer_name", ""),
            "Buyer Address": buyer_address,
            "Buyer State": values.get("buyer_state", ""),
            "Buyer GSTIN": values.get("buyer_gstin", ""),
            "Consignee Address (Shipped To)": values.get("consignee_address", ""),
            "Item Descriptions": item_summary["descriptions_joined"],
            "HSN/SAC Codes": item_summary["hsn_joined"],
            "Taxable Value (Total Before GST)": values.get("total_before_gst", item_summary["computed_taxable_total"]),
            "CGST %": values.get("cgst_pct", ""),
            "CGST Amount": values.get("cgst_amount", ""),
            "SGST %": values.get("sgst_pct", ""),
            "SGST Amount": values.get("sgst_amount", ""),
            "IGST %": values.get("igst_pct", ""),
            "IGST Amount": values.get("igst_amount", ""),
            "Total Amount After Tax": values.get("total_after_tax", ""),
            "Amount In Words": values.get("amount_in_words", ""),
            "Reverse Charge": values.get("reverse_charge_mark", ""),
            "Bank Name": static.get("bank_name", ""),
            "Bank Account No": static.get("bank_account_no", ""),
            "Bank Branch": static.get("bank_branch", ""),
            "Bank IFSC": static.get("bank_ifsc", ""),
        }

        review_reasons = []
        gstin_val = record.get("Buyer GSTIN") or ""
        if gstin_val and not fp.GSTIN_PATTERN.match(gstin_val):
            review_reasons.append("Buyer GSTIN does not match the standard 15-character GSTIN pattern")
        if not record.get("Invoice No"):
            review_reasons.append("Invoice No could not be read")
        if not record.get("Invoice Date"):
            review_reasons.append("Invoice Date could not be read")
        if item_summary["row_count"] == 0:
            review_reasons.append("No line items were detected in the items table")
        if record.get("Total Amount After Tax") in ("", None):
            review_reasons.append("Total Amount After Tax could not be read")
        low_conf_fields = [f.name for f in fields_out if f.confidence < LOW_CONFIDENCE_THRESHOLD]
        if low_conf_fields:
            review_reasons.append(f"Low OCR confidence on: {', '.join(low_conf_fields)}")

        record["Needs Review"] = "Yes" if review_reasons else "No"

        return ExtractionReport(
            record=record,
            fields=fields_out,
            needs_review=bool(review_reasons),
            review_reasons=review_reasons,
        )
