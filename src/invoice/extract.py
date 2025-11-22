
import re
import decimal
from typing import Dict, Optional


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
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(".", "")
        cleaned = cleaned.replace(",", ".")
    try:
        return decimal.Decimal(cleaned)
    except Exception:
        return None



SWED_INVOICE_NO_RE = re.compile(r"OCR[-/ ]*fakturanummer[:\s]*([0-9]{6,})", re.IGNORECASE)
SWED_FAKTURADATUM_RE = re.compile(r"Fakturadatum[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})", re.IGNORECASE)
SWED_SUBTOTAL_RE = re.compile(r"Summa\s+exkl\s+moms\s+([\d\s]+[,.]\d{2})\s*kr", re.IGNORECASE)
SWED_TAX_RE = re.compile(r"Moms\s*\(\s*(\d+)\s*%\s*\)\s+([\d\s]+[,.]\d{2})\s*kr", re.IGNORECASE)
SWED_TOTAL_RE = re.compile(r"Totalt\s+belopp\s+([\d\s]+[,.]\d{2})\s*kr", re.IGNORECASE)



def fallback_extract_swedish_invoice(text: str) -> Dict[str, Optional[str]]:
    """
    Heuristic extractor for Swedish invoices like the GodEl example.
    Returns simple strings (not Decimals) to backfill missing template values.
    """
    fields: Dict[str, Optional[str]] = {
        "invoice_no": None,
        "date": None,
        "subtotal": None,
        "tax": None,
        "total": None,
        "tax_rate": None,
    }

    m = SWED_INVOICE_NO_RE.search(text)
    if m:
        fields["invoice_no"] = m.group(1).strip()

    m = SWED_FAKTURADATUM_RE.search(text)
    if m:
        fields["date"] = m.group(1).strip()

    m = SWED_SUBTOTAL_RE.search(text)
    if m:
        fields["subtotal"] = m.group(1).strip()

    m = SWED_TAX_RE.search(text)
    if m:
        rate_str, tax_str = m.group(1).strip(), m.group(2).strip()
        fields["tax"] = tax_str
        try:
            fields["tax_rate"] = str(int(rate_str) / 100.0)
        except Exception:
            pass

    m = SWED_TOTAL_RE.search(text)
    if m:
        fields["total"] = m.group(1).strip()

    return fields


def extract_fields(text: str, template: Dict) -> Dict:
    rx = template.get("regex", {})

    data = {
        "invoice_no": _find(rx.get("invoice_no", ""), text),
        "date": _find(rx.get("date", ""), text),
        "subtotal": parse_amount(_find(rx.get("subtotal", ""), text)),
        "tax": parse_amount(_find(rx.get("tax", ""), text)),
        "total": parse_amount(_find(rx.get("total", ""), text)),
        "tax_rate": None,  # IMPORTANT: initialize so we don't get KeyError
    }

    # … math consistency & tax_rate calculation (if you want template-based logic here) …

    fallback = fallback_extract_swedish_invoice(text)

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

    out = {
        "invoice_no": data["invoice_no"],
        "date": data["date"],
        "subtotal": str(data["subtotal"]) if data["subtotal"] is not None else None,
        "tax": str(data["tax"]) if data["tax"] is not None else None,
        "total": str(data["total"]) if data["total"] is not None else None,
        "tax_rate": str(data["tax_rate"]) if data["tax_rate"] is not None else None,
    }
    return out


