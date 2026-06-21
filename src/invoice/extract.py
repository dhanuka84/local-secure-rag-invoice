import re
import decimal
from typing import Dict, Optional
from src.invoice.field_formats import normalize_date_string, parse_amount_string, normalize_tax_rate_string

DEFAULT_REGEX = {
    "invoice_no": r"(?im)^\s*(?:invoice\s*(?:no|number|#)?|ocr[-/ ]*fakturanummer|faktura\s*(?:nr|nummer|#))\s*[:\-]?\s*([A-Za-z0-9\-]+)",
    "date": r"(?im)^\s*(?:date|fakturadatum)\s*[:\-]?\s*(\d{4}[-/.]\d{2}[-/.]\d{2}|\d{2}[-/.]\d{2}[-/.]\d{4})",
    "subtotal": r"(?im)^\s*(?:subtotal|summa\s+exkl\s+moms)\s*[:\-]?\s*(?:kr|\$)?\s*([0-9][0-9\s.,]*)",
    "tax": r"(?im)^\s*(?:tax|moms)(?:\s*\(\s*\d+\s*%\s*\))?\s*[:\-]?\s*(?:kr|\$)?\s*([0-9][0-9\s.,]*)",
    "total": r"(?im)^\s*(?:total|totalt\s+belopp)\s*[:\-]?\s*(?:kr|\$)?\s*([0-9][0-9\s.,]*)",
}


def _find(pattern: str, text: str) -> Optional[str]:
    if not pattern:
        return None
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    groups = [g for g in m.groups() if g]
    return groups[-1] if groups else m.group(0)


def parse_amount(s: Optional[str]) -> Optional[decimal.Decimal]:
    return parse_amount_string(s)


SWED_INVOICE_NO_RE = re.compile(r"OCR[-/ ]*fakturanummer[:\s]*([0-9]{6,})", re.IGNORECASE)
SWED_FAKTURADATUM_RE = re.compile(r"Fakturadatum[:\s]*([0-9]{4}[-/.][0-9]{2}[-/.][0-9]{2}|[0-9]{2}[-/.][0-9]{2}[-/.][0-9]{4})", re.IGNORECASE)
SWED_SUBTOTAL_RE = re.compile(r"Summa\s+exkl\s+moms\s+([\d\s]+[,.]\d{2})\s*kr", re.IGNORECASE)
SWED_TAX_RE = re.compile(r"Moms\s*\(\s*(\d+)\s*%\s*\)\s+([\d\s]+[,.]\d{2})\s*kr", re.IGNORECASE)
SWED_TOTAL_RE = re.compile(r"Totalt\s+belopp\s+([\d\s]+[,.]\d{2})\s*kr", re.IGNORECASE)


def fallback_extract_swedish_invoice(text: str) -> Dict[str, Optional[str]]:
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

    # 2) Summary block totals row (correct 172,00 kr result)
    if not fields["total"]:
        totals_row = re.search(
            r"\b\d{4}-\d{2}-\d{2}\b\s+[\d\s]+[,.]\d{2}\s*kr\s+[\d\s]+[,.]\d{2}\s*kr\s+[\d\s]+[,.]\d{2}\s*kr\s+([\d\s]+[,.]\d{2})\s*kr",
            text,
            re.IGNORECASE,
        )
        if totals_row:
            fields["total"] = totals_row.group(1).strip()

    return fields


def fallback_extract_english_invoice(text: str) -> Dict[str, Optional[str]]:
    fields: Dict[str, Optional[str]] = {
        "invoice_no": None,
        "date": None,
        "subtotal": None,
        "tax": None,
        "total": None,
        "tax_rate": None,
    }

    patterns = {
        "invoice_no": re.compile(r"Invoice\s*(?:#|No|Number)?\s*[:\-]?\s*([A-Za-z0-9\-]+)", re.IGNORECASE),
        "date": re.compile(r"Date\s*[:\-]?\s*(\d{4}[-/.]\d{2}[-/.]\d{2}|\d{2}[-/.]\d{2}[-/.]\d{4})", re.IGNORECASE),
        "subtotal": re.compile(r"Subtotal\s*[:\-]?\s*\$?\s*([0-9][0-9,]*\.\d{2})", re.IGNORECASE),
        "tax": re.compile(r"Tax\s*[:\-]?\s*\$?\s*([0-9][0-9,]*\.\d{2})", re.IGNORECASE),
        "total": re.compile(r"Total\s*[:\-]?\s*\$?\s*([0-9][0-9,]*\.\d{2})", re.IGNORECASE),
    }

    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            fields[key] = match.group(1).strip()

    if fields["subtotal"] and fields["tax"] and not fields["tax_rate"]:
        subtotal = parse_amount(fields["subtotal"])
        tax = parse_amount(fields["tax"])
        if subtotal not in (None, decimal.Decimal("0")) and tax is not None:
            try:
                fields["tax_rate"] = str((tax / subtotal).quantize(decimal.Decimal("0.0001")))
            except Exception:
                fields["tax_rate"] = None

    return fields



def extract_fields(text: str, template: Dict) -> Dict:
    rx = dict(DEFAULT_REGEX)
    rx.update(template.get("regex", {}))

    data = {
        "invoice_no": _find(rx.get("invoice_no", ""), text),
        "date": normalize_date_string(_find(rx.get("date", ""), text)),
        "subtotal": parse_amount(_find(rx.get("subtotal", ""), text)),
        "tax": parse_amount(_find(rx.get("tax", ""), text)),
        "total": parse_amount(_find(rx.get("total", ""), text)),
        "tax_rate": None,  # IMPORTANT: initialize so we don't get KeyError
    }

    # … math consistency & tax_rate calculation (if you want template-based logic here) …

    fallback = fallback_extract_swedish_invoice(text)
    english_fallback = fallback_extract_english_invoice(text)

    if data["invoice_no"] is None:
        data["invoice_no"] = fallback.get("invoice_no") or english_fallback.get("invoice_no")

    if data["date"] is None:
        data["date"] = normalize_date_string(fallback.get("date") or english_fallback.get("date"))

    for key in ["subtotal", "tax", "total"]:
        if data[key] is None:
            raw_value = fallback.get(key) or english_fallback.get(key)
            if raw_value:
                data[key] = parse_amount(raw_value)

    if data["tax_rate"] is None and (fallback.get("tax_rate") or english_fallback.get("tax_rate")):
        try:
            normalized_rate = normalize_tax_rate_string(str(fallback.get("tax_rate") or english_fallback.get("tax_rate")))
            data["tax_rate"] = decimal.Decimal(str(normalized_rate)) if normalized_rate is not None else None
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
