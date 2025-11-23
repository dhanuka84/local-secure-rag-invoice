import re
import decimal
from typing import Dict, Optional


# ---------------------------
# Small helpers
# ---------------------------

def _find(pattern: str, text: str) -> Optional[str]:
    if not pattern:
        return None
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    groups = [g for g in m.groups() if g]
    return groups[-1] if groups else m.group(0)


def parse_amount(s: Optional[str]) -> Optional[decimal.Decimal]:
    if not s:
        return None
    s = s.strip()
    cleaned = re.sub(r"[^0-9,.\-]", "", s)
    if not cleaned:
        return None

    # If both comma and dot exist, assume comma is thousands sep, dot is decimal (US style)
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    # If only comma exists, treat it as decimal separator (Swedish style)
    elif "," in cleaned:
        cleaned = cleaned.replace(".", "")
        cleaned = cleaned.replace(",", ".")

    try:
        return decimal.Decimal(cleaned)
    except Exception:
        return None


# ---------------------------
# GodEl / generic Swedish invoice heuristics
# ---------------------------

SWED_INVOICE_NO_RE = re.compile(
    r"OCR[-/ ]*fakturanummer[:\s]*([0-9]{6,})",
    re.IGNORECASE,
)

SWED_FAKTURADATUM_RE = re.compile(
    r"Fakturadatum[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})",
    re.IGNORECASE,
)

SWED_SUBTOTAL_RE = re.compile(
    r"Summa\s+exkl\s+moms\s+([\d\s]+[,.]\d{2})\s*kr",
    re.IGNORECASE,
)

SWED_TAX_RE = re.compile(
    r"Moms\s*\(\s*(\d+)\s*%\s*\)\s+([\d\s]+[,.]\d{2})\s*kr",
    re.IGNORECASE,
)

SWED_TOTAL_RE = re.compile(
    r"Totalt\s+belopp\s+([\d\s]+[,.]\d{2})\s*kr",
    re.IGNORECASE,
)


def fallback_extract_swedish_invoice(text: str) -> Dict[str, Optional[str]]:
    """
    Heuristic extractor for Swedish invoices like the GodEl example.
    Returns simple strings (not Decimals) to backfill missing template values.
    Canonical keys:
        invoice_no, date, subtotal, tax, total, tax_rate
    """
    fields: Dict[str, Optional[str]] = {
        "invoice_no": None,
        "date": None,
        "subtotal": None,
        "tax": None,
        "total": None,
        "tax_rate": None,
    }

    # Invoice No
    m = SWED_INVOICE_NO_RE.search(text)
    if m:
        fields["invoice_no"] = m.group(1).strip()

    # Invoice date
    m = SWED_FAKTURADATUM_RE.search(text)
    if m:
        fields["date"] = m.group(1).strip()

    # Subtotal
    m = SWED_SUBTOTAL_RE.search(text)
    if m:
        fields["subtotal"] = m.group(1).strip()

    # Tax + rate
    m = SWED_TAX_RE.search(text)
    if m:
        rate_str, tax_str = m.group(1).strip(), m.group(2).strip()
        fields["tax"] = tax_str
        try:
            fields["tax_rate"] = str(int(rate_str) / 100.0)
        except Exception:
            pass

    # 1) Simple "Totalt belopp 172,00 kr"
    m = SWED_TOTAL_RE.search(text)
    if m:
        fields["total"] = m.group(1).strip()

    # 2) Summary block totals row (e.g. "2025-11-28 137,25 kr 34,31 kr 0,44 kr 172,00 kr")
    if not fields["total"]:
        totals_row = re.search(
            r"\b\d{4}-\d{2}-\d{2}\b\s+[\d\s]+[,.]\d{2}\s*kr\s+[\d\s]+[,.]\d{2}\s*kr\s+"
            r"[\d\s]+[,.]\d{2}\s*kr\s+([\d\s]+[,.]\d{2})\s*kr",
            text,
            re.IGNORECASE,
        )
        if totals_row:
            fields["total"] = totals_row.group(1).strip()

    return fields


# ---------------------------
# Täby-specific heuristics
# ---------------------------

def _safe_amount_str(value: Optional[str]) -> Optional[str]:
    """
    Normalize decimal string to 'X,YY' (no 'kr' here).
    The rest of the pipeline can then format as 'X,YY kr' for Swedish.
    """
    if not value:
        return None
    cleaned = value.replace(" ", "").replace(",", ".")
    try:
        amount = decimal.Decimal(cleaned)
        s = f"{amount:.2f}"
        whole, dec = s.split(".")
        return f"{whole},{dec}"
    except Exception:
        return None


def looks_like_taby_invoice(text: str) -> bool:
    """
    Täby invoices contain very reliable identifiers:
    - 'TÄBY KOMMUN'
    - 'Att betala i SEK'
    - 'Fakturanr'
    """
    t = text.upper()
    return (
        "TÄBY KOMMUN" in t
        or "ATT BETALA I SEK" in t
        or "FAKTURANR" in t
    )


TABY_DATE_RE = re.compile(r"\bDatum\s+([0-9]{4}-[0-9]{2}-[0-9]{2})", re.IGNORECASE)
TABY_INVOICE_NO_RE = re.compile(r"Fakturanr[:\s]*([0-9]{6,})", re.IGNORECASE)
TABY_OCR_RE = re.compile(r"OCR[- ]*refnr[:\s]*([0-9]{6,})", re.IGNORECASE)

TABY_TOTAL_RE = re.compile(
    r"Att\s+betala\s+i\s+SEK[:\s]*([0-9]+[.,]\d{2})",
    re.IGNORECASE,
)

TABY_MOMS_RE = re.compile(
    r"\bMoms\s+([0-9]+[.,]\d{2})",
    re.IGNORECASE,
)

TABY_SUBTOTAL_RE = re.compile(
    r"0%\s+([0-9]+[.,]\d{2})",
    re.IGNORECASE,
)


def fallback_extract_taby_invoice(text: str) -> Dict[str, Optional[str]]:
    """
    Täby invoice heuristic.

    IMPORTANT: returns canonical keys so the rest of the pipeline keeps working:
        invoice_no, date, subtotal, tax, total, tax_rate
    """
    fields: Dict[str, Optional[str]] = {
        "invoice_no": None,
        "date": None,
        "subtotal": None,
        "tax": None,
        "total": None,
        "tax_rate": None,
    }

    # Date
    m = TABY_DATE_RE.search(text)
    if m:
        fields["date"] = m.group(1).strip()

    # Invoice number (OCR-first, then Fakturanr)
    m = TABY_OCR_RE.search(text)
    if m:
        fields["invoice_no"] = m.group(1).strip()
    else:
        m = TABY_INVOICE_NO_RE.search(text)
        if m:
            fields["invoice_no"] = m.group(1).strip()

    # Total
    m = TABY_TOTAL_RE.search(text)
    if m:
        fields["total"] = _safe_amount_str(m.group(1))

    # Moms
    m = TABY_MOMS_RE.search(text)
    if m:
        fields["tax"] = _safe_amount_str(m.group(1))

    # Subtotal (0% line, if present)
    m = TABY_SUBTOTAL_RE.search(text)
    if m:
        fields["subtotal"] = _safe_amount_str(m.group(1))

    # Tax rate (rough heuristic)
    if "25.00% moms" in text:
        fields["tax_rate"] = "0.25"
    elif "12.00%" in text:
        fields["tax_rate"] = "0.12"
    elif "6.00%" in text:
        fields["tax_rate"] = "0.06"

    return fields


# ---------------------------
# Main entry: extract_fields
# ---------------------------

def extract_fields(text: str, template: Dict) -> Dict:
    # Ensure template is dict-like
    if not isinstance(template, dict):
        template = {}

    rx = template.get("regex", {})

    data = {
        "invoice_no": _find(rx.get("invoice_no", ""), text),
        "date": _find(rx.get("date", ""), text),
        "subtotal": parse_amount(_find(rx.get("subtotal", ""), text)),
        "tax": parse_amount(_find(rx.get("tax", ""), text)),
        "total": parse_amount(_find(rx.get("total", ""), text)),
        "tax_rate": None,
    }

    # Basic math consistency & tax_rate calculation (template-based)
    sub, tax, tot = data["subtotal"], data["tax"], data["total"]

    if sub is not None and tax is not None and tot is None:
        tot = sub + tax
        data["total"] = tot

    if sub is not None and tot is not None and tax is None:
        tax = tot - sub
        data["tax"] = tax

    if sub is not None and tax is not None and tot is not None:
        calc = sub + tax
        if (tot - calc).copy_abs() > decimal.Decimal("0.01"):
            tot = calc
            data["total"] = tot

    if sub not in (None, decimal.Decimal(0)) and tax is not None:
        try:
            data["tax_rate"] = (tax / sub).quantize(decimal.Decimal("0.0001"))
        except Exception:
            data["tax_rate"] = None
    else:
        data["tax_rate"] = None

    # ---------------------------
    # Choose appropriate fallback
    # ---------------------------
    if looks_like_taby_invoice(text):
        fallback = fallback_extract_taby_invoice(text)
    else:
        fallback = fallback_extract_swedish_invoice(text)

    # Backfill from fallback
    if data["invoice_no"] is None and fallback.get("invoice_no"):
        data["invoice_no"] = fallback["invoice_no"]

    if data["date"] is None and fallback.get("date"):
        data["date"] = fallback["date"]

    for key in ["subtotal", "tax", "total"]:
        if data[key] is None and fallback.get(key):
            data[key] = parse_amount(fallback[key])

    if data["tax_rate"] is None and fallback.get("tax_rate"):
        try:
            data["tax_rate"] = decimal.Decimal(str(fallback["tax_rate"]))
        except Exception:
            data["tax_rate"] = None

    # Normalize output as strings (canonical keys).
    # Your Swedish formatting / vision layer will:
    #   - map to localized keys (e.g. "Summa exkl moms")
    #   - replace dots with commas
    #   - append " kr" or "%" where needed.
    out = {
        "invoice_no": data["invoice_no"],
        "date": data["date"],
        "subtotal": str(data["subtotal"]) if data["subtotal"] is not None else None,
        "tax": str(data["tax"]) if data["tax"] is not None else None,
        "total": str(data["total"]) if data["total"] is not None else None,
        "tax_rate": str(data["tax_rate"]) if data["tax_rate"] is not None else None,
    }
    return out
