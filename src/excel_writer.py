"""
excel_writer.py
----------------
Appends one invoice's extracted data as a new row to a persistent Excel
file. Designed so running the pipeline N times on N different invoice
photos builds up an N-row sheet, and re-running it never rewrites,
reorders, or deletes rows that are already there.

Safety properties:
  - If the file doesn't exist yet: create it with the header row.
  - If it exists: open it, go to the first fully empty row after the
    existing data, and write there. Existing cells are never touched.
  - Column order is fixed by COLUMNS below, independent of dict key order.
  - A simple advisory lock file avoids two runs writing at the same time.
"""

import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List

from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter

COLUMNS: List[str] = [
    "Source File",
    "Processed At",
    "Invoice No",
    "Invoice Date",
    "Supplier Name",
    "Supplier Address",
    "Supplier GSTIN",
    "Buyer Name",
    "Buyer Address",
    "Buyer State",
    "Buyer GSTIN",
    "Consignee Address (Shipped To)",
    "Item Descriptions",
    "HSN/SAC Codes",
    "Taxable Value (Total Before GST)",
    "CGST %",
    "CGST Amount",
    "SGST %",
    "SGST Amount",
    "IGST %",
    "IGST Amount",
    "Total Amount After Tax",
    "Amount In Words",
    "Reverse Charge",
    "Bank Name",
    "Bank Account No",
    "Bank Branch",
    "Bank IFSC",
    "Needs Review",
]


@contextmanager
def _file_lock(path: str, timeout: float = 15.0):
    """Minimal cross-process advisory lock using a .lock sidecar file."""
    dir_path = os.path.dirname(os.path.abspath(path))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    lock_path = path + ".lock"
    start = time.time()
    while os.path.exists(lock_path):
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Could not acquire lock for {path} (another run may be in progress). "
                f"If you're sure nothing else is writing to it, delete {lock_path}."
            )
        time.sleep(0.2)
    open(lock_path, "w").close()
    try:
        yield
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def _ensure_workbook(path: str):
    dir_path = os.path.dirname(os.path.abspath(path))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(path):
        wb = load_workbook(path)
        ws = wb.active
        existing_header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        if existing_header != COLUMNS:
            raise ValueError(
                "Existing Excel file's header row does not match this pipeline's "
                "expected columns. Refusing to append to avoid misaligned data. "
                f"\nExpected: {COLUMNS}\nFound:    {existing_header}"
            )
        return wb, ws
    wb = Workbook()
    ws = wb.active
    ws.title = "Invoices"
    ws.append(COLUMNS)
    for i, col in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = max(14, min(40, len(col) + 4))
    return wb, ws


def append_invoice_row(excel_path: str, record: Dict) -> int:
    """
    Appends `record` (a dict keyed by field name -- see build_record() in
    extractor.py) as a new row. Missing keys are written as blank cells.
    Returns the 1-indexed row number the data was written to.
    """
    record = dict(record)
    record.setdefault("Processed At", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    with _file_lock(excel_path):
        wb, ws = _ensure_workbook(excel_path)
        next_row = ws.max_row + 1
        row_values = [record.get(col, "") for col in COLUMNS]
        ws.append(row_values)
        wb.save(excel_path)
        return next_row


def read_all_rows(excel_path: str) -> List[Dict]:
    """Convenience reader, mainly for tests/inspection."""
    if not os.path.exists(excel_path):
        return []
    wb = load_workbook(excel_path)
    ws = wb.active
    rows = []
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    for r in ws.iter_rows(min_row=2, values_only=True):
        if all(v in (None, "") for v in r):
            continue
        rows.append(dict(zip(header, r)))
    return rows
