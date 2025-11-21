from typing import Any, Dict, List, Tuple
import os

import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Lazy-loaded LayoutLM model so the project can run without it configured.
_processor = None
_model = None
_id2label = None
_LAYOUTLM_AVAILABLE = False


def _load_layoutlm():
    global _processor, _model, _id2label, _LAYOUTLM_AVAILABLE
    if _processor is not None:
        return

    model_id = os.getenv("LAYOUTLM_MODEL_ID")
    processor_id = os.getenv("LAYOUTLM_PROCESSOR_ID", model_id)

    if not model_id:
        print("[layoutlm_extract] LAYOUTLM_MODEL_ID not set; skipping LayoutLM extraction.")
        _LAYOUTLM_AVAILABLE = False
        return

    try:
        print(f"[layoutlm_extract] Loading LayoutLM model: {model_id}")
        _processor = LayoutLMv3Processor.from_pretrained(processor_id)
        _model = LayoutLMv3ForTokenClassification.from_pretrained(model_id).eval()
        _id2label = _model.config.id2label
        _LAYOUTLM_AVAILABLE = True
    except Exception as e:
        print(f"[layoutlm_extract] Failed to load LayoutLM model '{model_id}': {e}")
        _LAYOUTLM_AVAILABLE = False


def _normalize_bbox(token: Dict[str, Any], page_width: int, page_height: int) -> List[int]:
    x0 = token["x"]
    y0 = token["y"]
    x1 = token["x"] + token["width"]
    y1 = token["y"] + token["height"]
    return [
        int(1000 * x0 / page_width),
        int(1000 * y0 / page_height),
        int(1000 * x1 / page_width),
        int(1000 * y1 / page_height),
    ]


def _run_layoutlm_on_pages(pages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    global _processor, _model, _id2label
    _load_layoutlm()
    if not _LAYOUTLM_AVAILABLE:
        return [], []

    all_tokens: List[Dict[str, Any]] = []
    words: List[str] = []
    boxes: List[List[int]] = []

    for page in pages:
        w, h = page["width"], page["height"]
        for token in page["tokens"]:
            all_tokens.append(token)
            words.append(token["text"])
            boxes.append(_normalize_bbox(token, w, h))

    if not words:
        return [], []

    encoding = _processor(
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    with torch.no_grad():
        outputs = _model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()

    labels = [_id2label[p] for p in predictions]
    return all_tokens, labels


def _normalize_amount(raw: str):
    if not raw:
        return None
    s = raw.replace(" ", "").replace(",", ".")
    for cc in ["€", "£", "CHF"]:
        s = s.replace(cc, "")
    keep = [ch for ch in s if (ch.isdigit() or ch == ".")]
    if not keep:
        return None
    try:
        return float("".join(keep))
    except ValueError:
        return None


def _normalize_rate(raw: str):
    if not raw:
        return None
    s = raw.replace(" ", "").replace("%", "")
    try:
        v = float(s.replace(",", "."))
    except ValueError:
        return None
    return v / 100.0 if v > 1.0 else v


def _extract_fields_and_line_items(
    tokens: List[Dict[str, Any]],
    labels: List[str],
):
    header_fields = {
        "invoice_id": [],
        "invoice_date": [],
        "vendor_name": [],
        "total_amount": [],
        "vat_amount": [],
        "vat_rate": [],
    }

    line_items: List[Dict[str, Any]] = []
    current_row = {"qty": [], "unit_price": [], "tax_rate": [], "line_total": []}

    for token, label in zip(tokens, labels):
        text = token["text"]

        # Header-level fields
        if "INVOICE_ID" in label:
            header_fields["invoice_id"].append(text)
        elif "INVOICE_DATE" in label:
            header_fields["invoice_date"].append(text)
        elif "VENDOR_NAME" in label:
            header_fields["vendor_name"].append(text)
        elif "TOTAL_AMOUNT" in label:
            header_fields["total_amount"].append(text)
        elif "VAT_AMOUNT" in label:
            header_fields["vat_amount"].append(text)
        elif "VAT_RATE" in label:
            header_fields["vat_rate"].append(text)

        # Line items (simplified row handling)
        elif "LINE_QTY" in label:
            current_row["qty"].append(text)
        elif "LINE_UNIT_PRICE" in label:
            current_row["unit_price"].append(text)
        elif "LINE_TAX_RATE" in label:
            current_row["tax_rate"].append(text)
        elif "LINE_TOTAL" in label:
            current_row["line_total"].append(text)
            line_items.append(
                {
                    "quantity_raw": " ".join(current_row["qty"]),
                    "unit_price_raw": " ".join(current_row["unit_price"]),
                    "tax_rate_raw": " ".join(current_row["tax_rate"]),
                    "line_total_raw": " ".join(current_row["line_total"]),
                }
            )
            current_row = {"qty": [], "unit_price": [], "tax_rate": [], "line_total": []}

    header = {
        "invoice_id": " ".join(header_fields["invoice_id"]) or None,
        "invoice_date": " ".join(header_fields["invoice_date"]) or None,
        "vendor_name": " ".join(header_fields["vendor_name"]) or None,
        "total_amount": _normalize_amount(" ".join(header_fields["total_amount"])),
        "vat_amount": _normalize_amount(" ".join(header_fields["vat_amount"])),
        "vat_rate": _normalize_rate(" ".join(header_fields["vat_rate"])),
    }

    for item in line_items:
        item["quantity"] = _normalize_amount(item["quantity_raw"])
        item["unit_price"] = _normalize_amount(item["unit_price_raw"])
        item["tax_rate"] = _normalize_rate(item["tax_rate_raw"])
        item["line_total"] = _normalize_amount(item["line_total_raw"])

    return header, line_items


def extract_with_layoutlm(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    tokens, labels = _run_layoutlm_on_pages(pages)
    if not tokens or not labels:
        return {"header": {}, "line_items": []}
    header, line_items = _extract_fields_and_line_items(tokens, labels)
    return {"header": header, "line_items": line_items}
