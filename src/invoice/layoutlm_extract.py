import os
from typing import Any, Dict, List, Tuple

from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import AutoModelForTokenClassification, LayoutLMv3Processor

MODEL_ID = os.getenv("LAYOUTLM_MODEL_ID")
PROCESSOR_ID = os.getenv("LAYOUTLM_PROCESSOR_ID", MODEL_ID or "")
DEVICE_NAME = os.getenv("LAYOUTLM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE_NAME)

_processor: LayoutLMv3Processor | None = None
_model: AutoModelForTokenClassification | None = None
_available = None


def _pdf_to_first_page_image(pdf_path: str) -> Image.Image:
    pages = convert_from_path(pdf_path, first_page=1, last_page=1)
    if not pages:
        raise ValueError(f"No pages found in PDF: {pdf_path}")
    return pages[0].convert("RGB")


def _load_layoutlm() -> bool:
    global _processor, _model, _available
    if _available is not None:
        return _available

    if not MODEL_ID:
        print("[layoutlm_extract] LAYOUTLM_MODEL_ID not set; skipping LayoutLM extraction.")
        _available = False
        return False

    try:
        _processor = LayoutLMv3Processor.from_pretrained(PROCESSOR_ID or MODEL_ID)
        _model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
        _model.to(DEVICE)
        _model.eval()
        _available = True
        print(f"[layoutlm_extract] Loaded {MODEL_ID} on {DEVICE}.")
        return True
    except Exception as exc:
        print(f"[layoutlm_extract] Failed to load LayoutLM model '{MODEL_ID}': {exc}")
        _processor = None
        _model = None
        _available = False
        return False


def _join(buf: List[str]) -> str | None:
    if not buf:
        return None
    joined = "".join(buf)
    return joined.replace("##", "")


def _tokens_to_fields(tokens: List[str], labels: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fields = {
        "invoice_no": None,
        "date": None,
        "subtotal": None,
        "tax": None,
        "total": None,
        "tax_rate": None,
    }
    buffers = {
        "INVOICE_NO": [],
        "DATE": [],
        "SUBTOTAL": [],
        "TAX": [],
        "TOTAL": [],
        "TAX_RATE": [],
    }

    for token, label in zip(tokens, labels):
        if not (label.startswith("B-") or label.startswith("I-")):
            continue
        key = label.split("-", 1)[1]
        if key in buffers:
            buffers[key].append(token)

    fields["invoice_no"] = _join(buffers["INVOICE_NO"])
    fields["date"] = _join(buffers["DATE"])
    fields["subtotal"] = _join(buffers["SUBTOTAL"])
    fields["tax"] = _join(buffers["TAX"])
    fields["total"] = _join(buffers["TOTAL"])
    fields["tax_rate"] = _join(buffers["TAX_RATE"])

    return fields, []


def extract_with_layoutlm(pdf_path: str) -> Dict[str, Any]:
    if not _load_layoutlm():
        return {"fields": {}, "line_items": [], "raw_tokens": [], "device": str(DEVICE)}

    assert _processor is not None
    assert _model is not None

    try:
        image = _pdf_to_first_page_image(pdf_path)
        encoding = _processor(
            image,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        encoding = {key: value.to(DEVICE) for key, value in encoding.items()}

        with torch.inference_mode():
            outputs = _model(**encoding)
            predictions = outputs.logits.argmax(-1)[0].detach().cpu().tolist()

        token_ids = encoding["input_ids"][0].detach().cpu().tolist()
        tokens = _processor.tokenizer.convert_ids_to_tokens(token_ids)
        labels = [str(_model.config.id2label[p]) for p in predictions]
        fields, line_items = _tokens_to_fields(tokens, labels)
        raw_tokens = [{"token": token, "label": label} for token, label in zip(tokens, labels)]
        return {
            "fields": fields,
            "line_items": line_items,
            "raw_tokens": raw_tokens,
            "device": str(DEVICE),
        }
    except Exception as exc:
        print(f"[layoutlm_extract] Extraction failed for '{pdf_path}': {exc}")
        return {"fields": {}, "line_items": [], "raw_tokens": [], "device": str(DEVICE)}
