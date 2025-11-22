from __future__ import annotations
from typing import Dict, Any, Optional, List
import re
import spacy

# Load Swedish spaCy model once
NLP = spacy.load("sv_core_news_lg")


def _best_org(doc) -> Optional[str]:
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
    if not orgs:
        return None
    return max(orgs, key=len)


def _all_dates(doc) -> List[str]:
    return [ent.text.strip() for ent in doc.ents if ent.label_ == "DATE"]


def _find_invoice_number(text: str) -> Optional[str]:
    kw = re.compile(
        r"(ocr|faktura\s*nr|fakturanummer|faktura\s*nummer)\s*[:#]?\s*([0-9]+)",
        re.IGNORECASE,
    )
    for m in kw.finditer(text):
        return m.group(2)
    raw = re.findall(r"\b\d{10,15}\b", text)
    return raw[0] if raw else None


def _find_total_amount(text: str) -> Optional[str]:
    amount_re = re.compile(r"\d[\d\s]*[.,]\d{2}")
    keywords = ["att betala", "belopp att betala", "summa att betala"]
    for line in text.splitlines():
        lower = line.lower()
        if any(kw in lower for kw in keywords):
            amts = amount_re.findall(line)
            if amts:
                return amts[-1]
    return None


def extract_with_spacy(text: str) -> Dict[str, Any]:
    doc = NLP(text)

    return {
        "vendor": _best_org(doc),
        "dates": _all_dates(doc),
        "invoice_no": _find_invoice_number(text),
        "total": _find_total_amount(text),
    }
