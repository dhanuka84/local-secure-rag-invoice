
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
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return decimal.Decimal(s)
    except Exception:
        return None


def extract_fields(text: str, template: Dict) -> Dict:
    rx = template.get("regex", {})

    data = {
        "invoice_no": _find(rx.get("invoice_no", ""), text),
        "date": _find(rx.get("date", ""), text),
        "subtotal": parse_amount(_find(rx.get("subtotal", ""), text)),
        "tax": parse_amount(_find(rx.get("tax", ""), text)),
        "total": parse_amount(_find(rx.get("total", ""), text)),
    }

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

    out = {
        "invoice_no": data["invoice_no"],
        "date": data["date"],
        "subtotal": str(data["subtotal"]) if data["subtotal"] is not None else None,
        "tax": str(data["tax"]) if data["tax"] is not None else None,
        "total": str(data["total"]) if data["total"] is not None else None,
        "tax_rate": str(data["tax_rate"]) if data["tax_rate"] is not None else None,
    }
    return out
