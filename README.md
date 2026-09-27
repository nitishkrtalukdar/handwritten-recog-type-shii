# Handwritten Invoice → Excel Extraction

Extracts data from **hand-written Indian GST tax invoices** (photographed or
scanned) that follow a **fixed pre-printed template**, and appends each
invoice as one new row to a persistent Excel file — without ever touching
rows written by previous runs.

Built and validated against a real sample invoice (Bharat Associates tax
invoice) used to design this project.

---

## Why this architecture

Handwritten text recognition is a genuinely hard ML problem — not something
Tesseract (a printed-text OCR engine) handles well. Testing this directly
against the real sample invoice used to build this project:

| Region | Content type | Tesseract output |
|---|---|---|
| Supplier GSTIN (printed) | machine-printed | `GSTIN ; 09GBVPS8212J1ZP` correct |
| CGST amount (handwritten "625=00") | handwritten | `TIaC= Eb` garbage |

This confirms the core design decision: **two OCR engines, used for
different content**:

- **TrOCR** (`microsoft/trocr-base-handwritten`) — a transformer
  encoder-decoder model specifically fine-tuned for handwriting. Used for
  every field a person actually writes by hand.
- **Tesseract** — used only for machine-printed text, and only if you
  choose to re-verify it per invoice. In practice, this project instead
  stores a vendor's printed details (their own name/address/GSTIN/bank
  info) once as `static_fields` in the template, since those never change
  across that vendor's invoices — this is both more reliable and much
  faster than re-OCRing static print every single run.

Because the invoice has a **fixed template but a variable number of line
items**, field locations are defined as a **calibrated ROI (region of
interest) template** (`config/*.json`) with normalized (0-1) coordinates,
and the item table specifically uses a **row-segmentation algorithm**
(horizontal ink-projection profiling) to automatically detect how many rows
are actually filled in on a given invoice, rather than assuming a fixed
count.

---

## Project structure

```
handwritten_invoice_ocr/
├── main.py                        # CLI entry point -- run this per invoice
├── requirements.txt
├── config/
│   └── template_bharat_associates.json   # calibrated ROI template (sample vendor)
├── src/
│   ├── preprocess.py               # deskew / denoise / contrast normalize
│   ├── row_segmentation.py         # auto-detect variable item-table rows
│   ├── ocr_engine.py                # TrOCR (handwriting) + Tesseract (printed) wrappers
│   ├── field_parser.py              # GSTIN/date/amount/percent cleanup & validation
│   ├── extractor.py                 # orchestrates the full pipeline per invoice
│   └── excel_writer.py              # append-only Excel writer
├── tools/
│   └── calibrate_template.py       # interactive tool to calibrate a NEW vendor layout
└── output/
    └── invoices.xlsx                # created automatically on first run
```

---

## Setup

```bash
# System dependency (Tesseract binary -- not a pip package)
sudo apt-get install tesseract-ocr

# Python dependencies
pip install -r requirements.txt
```

The first time you actually run extraction, `transformers` will download the
TrOCR model weights (~1.3 GB) from huggingface.co -- this needs internet
access once; after that it's cached locally (`~/.cache/huggingface`) and
works fully offline.

> A GPU is optional but speeds things up a lot. CPU works fine for
> occasional/low-volume use -- expect a few seconds per invoice on CPU.

---

## Usage

```bash
python main.py --image path/to/invoice_photo.jpg \
                --template config/template_bharat_associates.json \
                --output output/invoices.xlsx
```

Run this once per invoice photo. Each run:
1. Preprocesses the image (fixes rotation, enhances contrast).
2. Crops every field region from the template.
3. OCRs each region with the right engine.
4. Cleans/validates the text (GSTIN format, dates, currency amounts).
5. Auto-detects and reads however many item rows are actually filled in.
6. Appends **one new row** to the Excel file. If the file doesn't exist
   yet, it's created with the header row first.

Add `--show-fields` to print every field's raw OCR text, parsed value, and
confidence score -- useful while checking accuracy on new samples:

```bash
python main.py --image invoice2.jpg --show-fields
```

If any field looks unreliable (unreadable GSTIN, missing invoice number,
zero item rows detected, or low OCR confidence on any field), the row is
still written, but the **"Needs Review"** column is set to `Yes` and the
script exits with status code 2 -- convenient for flagging rows in a batch
script without silently trusting bad data.

### Using this on a different vendor's invoice layout

The included template is calibrated specifically to one vendor's printed
stationery (field positions, table columns, etc. are all specific to that
physical layout). For a different vendor's invoice book:

```bash
python tools/calibrate_template.py --image path/to/one_sample.jpg \
                                    --out config/template_myvendor.json
```

This opens an interactive window (needs a display -- run it on your own
desktop, not a headless server) where you drag a box around each field
once. It also asks you to type in the vendor's fixed printed details (name,
address, GSTIN, bank info) once, since those don't need to be re-OCR'd on
every invoice.

---

## Excel output schema

One row per invoice, columns (in order):

`Source File, Processed At, Invoice No, Invoice Date, Supplier Name,
Supplier Address, Supplier GSTIN, Buyer Name, Buyer Address, Buyer State,
Buyer GSTIN, Consignee Address (Shipped To), Item Descriptions, HSN/SAC
Codes, Taxable Value (Total Before GST), CGST %, CGST Amount, SGST %, SGST
Amount, IGST %, IGST Amount, Total Amount After Tax, Amount In Words,
Reverse Charge, Bank Name, Bank Account No, Bank Branch, Bank IFSC, Needs
Review`

This maps directly onto the GST mandatory-invoice-field checklist (basic
details, supplier/buyer info, classification codes, tax breakdown, totals &
declarations, banking details). Multiple line items on one invoice are
summarized into a single `Item Descriptions` cell (e.g. `1) Stamp Paper
Value (Rate: 21980); 2) Notary Charges (Rate: 2180); 3) Service Charges
(Rate: 6935)`) since each Excel row represents one invoice, not one line
item.

**Append-only guarantee:** `excel_writer.py` opens the existing workbook,
finds the first blank row after existing data, and writes there -- it never
rewrites, reorders, or deletes prior rows. It also refuses to append if an
existing file's header row doesn't match this schema, to avoid silently
misaligning columns.

---

## What's been validated vs. what needs your own test run

Everything in this pipeline was tested against a real sample invoice
**except the actual TrOCR handwriting inference**, which needs to download
model weights from huggingface.co -- the sandbox this was built in doesn't
allow that host, so it could only be verified by architecture/unit testing
with a mock OCR engine (confirmed correct number of fields, correct row
detection count, correct record assembly, correct Excel append behavior).
Concretely validated on the real sample:

- Preprocessing pipeline runs cleanly, no distortion
- ROI coordinates calibrated and verified against the real layout
- Row-segmentation algorithm correctly found all 3 line items
- Tesseract correctly read printed regions (GSTIN, bank details)
- Tesseract confirmed unreliable on handwriting (motivating the TrOCR split)
- Field parser: GSTIN validation, date parsing, currency parsing (incl. the
  common Indian handwriting style `625=00` / `21,980/-`) -- two real bugs
  were caught and fixed during this testing
- Excel append-only behavior, multi-row integrity, header-mismatch safety check
- Full pipeline orchestration (mocked OCR) end-to-end

**On your machine, run a real invoice through `main.py --show-fields`
first** and check the printed confidence scores/raw text before trusting
the Excel output for anything important -- handwriting recognition accuracy
depends heavily on how legible the specific handwriting is.

## Known limitations / possible improvements

- **"Reverse Charge Yes/No"** is printed with the person circling one
  option by hand -- that's a mark-detection problem, not a text-reading
  problem. The current field just captures raw OCR text around that area;
  detecting which word is circled would need a small circle/ring-detection
  step (e.g. Hough circle transform) rather than OCR.
- **Multi-vendor use** requires calibrating one template per distinct
  printed layout (`tools/calibrate_template.py`). It does not (yet)
  auto-detect which vendor's template a given photo matches.
- **Accuracy on very messy handwriting**: `microsoft/trocr-base-handwritten`
  is trained mostly on English handwriting (IAM dataset lineage). For
  materially better accuracy on your specific handwriting samples, consider
  fine-tuning it on a small labeled set of your own invoices (a natural
  next step for an academic project -- the current pipeline structure is
  set up so swapping in a fine-tuned checkpoint is just a `--trocr-model`
  path change).
- Skew correction handles small rotations (<15 deg); a badly crooked photo
  should be retaken rather than relied on.
