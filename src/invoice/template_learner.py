
import json
from langchain_ollama import OllamaLLM

LLM = OllamaLLM(model="llama3.2")

PROMPT = """You are a senior parsing engineer.
Given this invoice text, propose JSON with robust regex patterns to extract fields:
invoice_no, date, subtotal, tax, total.
- Use permissive and case-insensitive labels.
- Capture numbers with commas/decimals.
- Dates may be YYYY-MM-DD or MM/DD/YYYY.
Return ONLY valid JSON with a top-level key "regex".
INVOICE_TEXT:
---
{invoice_text}
---"""


def learn_regexes(invoice_text: str) -> dict:
    sample = invoice_text[:4000]
    raw = LLM.invoke(PROMPT.format(invoice_text=sample))
    try:
        data = json.loads(raw)
        regex = data.get("regex", data)
    except Exception:
        regex = {
            "invoice_no": r"(?i)invoice\s*(?:no|#)\s*[:\-]?\s*([A-Za-z0-9\-]+)",
            "date": r"(?i)date\s*[:\-]?\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})",
            "subtotal": r"(?i)subtotal\s*[:\-]?\s*\$?([0-9,]+\.?\d{0,2})",
            "tax": r"(?i)tax(?:\s*(?:rate|amount))?\s*[:\-]?\s*\$?([0-9,]+\.?\d{0,2})",
            "total": r"(?i)total\s*[:\-]?\s*\$?([0-9,]+\.?\d{0,2})",
        }
    return {
        "version": 1,
        "vendor_hint": "",
        "anchors": {"subtotal": "Subtotal", "tax": "Tax", "total": "Total"},
        "regex": regex,
    }


REFINE_PROMPT = """You previously proposed these regex patterns for an invoice:

CURRENT_REGEX:
{current_regex_json}

When applied to the invoice text below, they extracted the WRONG value for field "{field_name}":
- Extracted: "{got}"
- Correct value: "{expected}"

INVOICE_TEXT:
---
{invoice_text}
---

Please FIX the regex patterns to correctly extract the field "{field_name}".
Return ONLY valid JSON with a top-level key "regex". Keep the other fields working if possible.
"""


def refine_regexes(
    invoice_text: str,
    current_template: dict,
    field_name: str,
    expected: str,
    got: str,
) -> dict:
    current_regex = current_template.get("regex", {})
    prompt = REFINE_PROMPT.format(
        current_regex_json=json.dumps(current_regex, indent=2),
        field_name=field_name,
        expected=expected,
        got=got,
        invoice_text=invoice_text[:4000],
    )
    raw = LLM.invoke(prompt)
    try:
        data = json.loads(raw)
        new_regex = data.get("regex", data)
    except Exception:
        new_regex = current_regex

    new_tmpl = dict(current_template)
    new_tmpl["regex"] = new_regex
    new_tmpl["version"] = int(new_tmpl.get("version", 1)) + 1
    new_tmpl["refined_field"] = field_name
    new_tmpl["refined_from"] = got
    new_tmpl["refined_to"] = expected
    return new_tmpl
