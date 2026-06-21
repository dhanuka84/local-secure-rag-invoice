import decimal
import re
from datetime import datetime
from typing import Optional

FIELD_FORMATS = {
    "date": [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%m.%d.%Y",
    ],
    "amount": [
        "1234.56",
        "1,234.56",
        "1234,56",
        "1 234,56",
        "1.234,56",
    ],
    "tax_rate": [
        "25%",
        "25.0%",
        "0.25",
        "25",
    ],
}


def normalize_date_string(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = value.strip().rstrip(".,;")
    for fmt in FIELD_FORMATS["date"]:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def parse_amount_string(value: Optional[str]) -> Optional[decimal.Decimal]:
    if not value:
        return None
    cleaned = re.sub(r"[^0-9,.\-]", "", value.strip())
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


def normalize_tax_rate_string(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = value.strip().rstrip(".,;")
    if raw.endswith("%"):
        raw = raw[:-1].strip()
        try:
            return str(decimal.Decimal(raw) / decimal.Decimal("100"))
        except Exception:
            return value
    try:
        dec = decimal.Decimal(raw.replace(",", "."))
        if dec > 1:
            dec = dec / decimal.Decimal("100")
        return str(dec)
    except Exception:
        return value


def normalize_field_value(field_name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if field_name == "date":
        return normalize_date_string(value)
    if field_name in {"subtotal", "tax", "total"}:
        amount = parse_amount_string(value)
        return str(amount) if amount is not None else value
    if field_name == "tax_rate":
        return normalize_tax_rate_string(value)
    return value.strip() if isinstance(value, str) else str(value)
