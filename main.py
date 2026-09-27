#!/usr/bin/env python3
"""
main.py
-------
Command-line entry point.

    python main.py --image path/to/invoice.jpg \
                    --template config/template_bharat_associates.json \
                    --output output/invoices.xlsx

Run it once per handwritten invoice photo. Each run appends exactly one new
row to the Excel file (creating it with headers on the very first run).
Existing rows are never modified.
"""

import argparse
import sys

from src.excel_writer import append_invoice_row
from src.extractor import InvoiceExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract a handwritten invoice into an Excel row.")
    parser.add_argument("--image", required=True, help="Path to the invoice photo/scan (jpg/png).")
    parser.add_argument(
        "--template",
        default="config/template_bharat_associates.json",
        help="Path to the ROI template JSON for this vendor's invoice layout.",
    )
    parser.add_argument(
        "--output",
        default="output/invoices.xlsx",
        help="Excel file to append this invoice's data to.",
    )
    parser.add_argument(
        "--trocr-model",
        default="microsoft/trocr-base-handwritten",
        help="HuggingFace model id for handwriting OCR. Use "
             "'microsoft/trocr-large-handwritten' for higher accuracy at the cost of speed.",
    )
    parser.add_argument(
        "--show-fields",
        action="store_true",
        help="Print every extracted field with its raw OCR text and confidence (debugging aid).",
    )
    args = parser.parse_args()

    from src.ocr_engine import TrOCREngine, TesseractEngine

    extractor = InvoiceExtractor(
        template_path=args.template,
        trocr_engine=TrOCREngine(model_name=args.trocr_model),
        tesseract_engine=TesseractEngine(),
    )

    print(f"Processing {args.image} ...")
    print("(First run will download the TrOCR model from huggingface.co -- "
          "this can take a few minutes and needs internet access.)")
    report = extractor.extract(args.image)

    if args.show_fields:
        print("\n--- extracted fields ---")
        for f in report.fields:
            flag = "  <-- LOW CONFIDENCE" if f.confidence < 35 else ""
            print(f"{f.name:28s} raw='{f.raw_text}'  parsed={f.value!r}  conf={f.confidence}{flag}")

    row_num = append_invoice_row(args.output, report.record)

    print(f"\nWrote invoice as row {row_num} in {args.output}")
    if report.needs_review:
        print("\n[!] NEEDS REVIEW -- please double check this row:")
        for reason in report.review_reasons:
            print(f"   - {reason}")
        sys.exit(2)  # non-zero exit so this is easy to catch in a batch script
    else:
        print("No review flags raised.")


if __name__ == "__main__":
    main()
