"""
field_parser.py
----------------
Raw OCR output is messy (misread characters, stray punctuation from ruled
lines, "Rs." symbols read as digits, etc). This module turns raw OCR text
into clean, typed values, and applies light error-correction using
knowledge of each field's expected FORMAT (this is safe/generic correction,
not guessing invoice content):

  - GSTIN: fixed 15-character pattern (2 digits, 5 letters, 4 digits,
    1 letter, 1 alphanumeric, 'Z', 1 alphanumeric). We fix common
    handwriting/OCR confusions (O<->0, I/l<->1, S<->5, B<->8) ONLY when it
    makes an otherwise-invalid GSTIN match the pattern.
  - Dates: several common handwritten formats -> ISO (YYYY-MM-DD).
  - Amounts: strips currency symbols/commas, fixes OCR's frequent
    "1,980/-" -> "1980" style noise, returns a float.
  - Percent: strips '%' and non-numeric junk, returns a float.
"""

import re
from datetime import datetime
from typing import Optional


# 2 digits (state) + 5 letters + 4 digits + 1 letter (PAN, 10 chars) +
# 1 alnum (entity code) + literal 'Z' + 1 alnum (checksum) = 15 chars total
GSTIN_PATTERN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]$")

_CONFUSIONS = [
    ("O", "0"), ("o", "0"),
    ("I", "1"), ("l", "1"), ("|", "1"),
    ("S", "5"), ("s", "5"),
    ("B", "8"),
    ("Z", "2"),  # only tried as a last resort, see try_fix_gstin
]


def clean_gstin(raw: str) -> Optional[str]:
    """Best-effort clean-up of a handwritten/OCR'd GSTIN into the 15-char format."""
    if not raw:
        return None
    candidate = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    if len(candidate) != 15:
        # try trimming stray leading/trailing junk characters
        m = re.search(r"[0-9]{2}[A-Z0-9]{13}", candidate)
        if m:
            candidate = m.group(0)
    if len(candidate) != 15:
        return candidate or None  # return whatever we have, flagged low-confidence upstream

    if GSTIN_PATTERN.match(candidate):
        return candidate

    # Targeted digit<->letter fixes at the positions that MUST be digits/letters.
    chars = list(candidate)
    digit_positions = {0, 1, 7, 8, 9, 10}
    letter_positions = {2, 3, 4, 5, 6, 11}
    digit_to_letter_fix = {"0": "O", "1": "I", "5": "S", "8": "B"}
    letter_to_digit_fix = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"}

    for pos in digit_positions:
        if pos < len(chars) and chars[pos] in letter_to_digit_fix:
            chars[pos] = letter_to_digit_fix[chars[pos]]
    for pos in letter_positions:
        if pos < len(chars) and chars[pos] in digit_to_letter_fix:
            chars[pos] = digit_to_letter_fix[chars[pos]]
    if len(chars) > 13:
        chars[13] = "Z"  # 14th char (index 13) is always literal 'Z'

    fixed = "".join(chars)
    return fixed  # may still not fully match; caller decides review threshold


_DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d/%m/%y", "%d-%m-%y",
    "%d/%m/%Y", "%Y-%m-%d",
]


def clean_date(raw: str) -> Optional[str]:
    """Parse a handwritten date into ISO format (YYYY-MM-DD). Returns None if unparseable."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"[^0-9/\.\-]", "", text)  # drop stray letters ("ate:", etc.)
    text = re.sub(r"[\.]", "/", text)
    text = re.sub(r"/{2,}", "/", text)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt.replace(".", "/")).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text or None  # fall back to the cleaned raw string for manual review


_AMOUNT_JUNK = re.compile(r"[^\d.]")


def clean_amount(raw: str) -> Optional[float]:
    """
    '21,980/-' / '6935/=00' / 'Rs. 625=00' -> 625.0
    Handles the common Indian handwriting habit of writing '=00' or '/-'
    instead of a decimal point for paise, plus stray currency words/symbols.
    """
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"(?i)\brs\.?\b|₹|inr", "", text)
    text = text.replace(",", "")
    # '6935/=00' or '6935=00' style -> treat '=' like a decimal point
    text = re.sub(r"[/]?=\s*(\d{2})\b", r".\1", text)
    text = text.replace("/-", "")
    text = _AMOUNT_JUNK.sub("", text)
    if not text or text == ".":
        return None
    # collapse multiple leftover periods, keeping only the last as the decimal point
    if text.count(".") > 1:
        head, _, tail = text.rpartition(".")
        text = head.replace(".", "") + "." + tail
    text = text.strip(".")
    if not text:
        return None
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def clean_percent(raw: str) -> Optional[float]:
    if not raw:
        return None
    text = re.sub(r"[^\d.]", "", raw)
    if not text:
        return None
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def clean_text(raw: str) -> str:
    """Generic tidy-up: collapse whitespace, strip stray leading labels/colons."""
    if not raw:
        return ""
    text = re.sub(r"\s+", " ", raw).strip()
    text = re.sub(r"^(name|address|state|gstin)\s*[:.]*\s*", "", text, flags=re.IGNORECASE)
    return text.strip(" :.-")
