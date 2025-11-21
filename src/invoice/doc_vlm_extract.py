import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from PIL import Image
from pdf2image import convert_from_path
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel


def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Render a PDF to a list of PIL.Image images (one per page)."""
    return convert_from_path(pdf_path)


MODEL_ID = os.getenv(
    "DOC_VLM_MODEL_ID",
    "naver-clova-ix/donut-base-finetuned-cord-v2",
)
MAX_LENGTH = int(os.getenv("DOC_VLM_MAX_LENGTH", "512"))
TASK_PROMPT = os.getenv("DOC_VLM_TASK_PROMPT", "<s_cord-v2>")


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_doc_vlm() -> Tuple[DonutProcessor, VisionEncoderDecoderModel]:
    processor = DonutProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.to(_get_device())
    model.eval()
    return processor, model


def extract_with_doc_vlm(pdf_path: str, page_index: int = 0) -> Dict[str, Any]:
    """Extract structured fields + line items using Donut (JSON mode)."""
    processor, model = _load_doc_vlm()
    device = _get_device()

    images = pdf_to_images(pdf_path)
    if not images:
        return {"fields": {}, "line_items": [], "raw": {}, "model_output": ""}

    img = images[min(max(page_index, 0), len(images) - 1)]
    image = img.convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    gen_kwargs = {
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
        "bad_words_ids": [[processor.tokenizer.unk_token_id]],
    }
    if MAX_LENGTH > 0:
        gen_kwargs["max_length"] = MAX_LENGTH
    else:
        gen_kwargs["max_length"] = model.decoder.config.max_position_embeddings

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            **gen_kwargs,
    )

    # For transformers generate(), outputs is usually a Tensor [batch, seq_len]
    # so we pass it directly to batch_decode
    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    try:
        json_data = processor.token2json(sequence)
    except Exception as e:
        print("[DOCVLM] token2json failed:", e)
        json_data = {}


    fields, line_items = _map_donut_json_to_fields(json_data)

    return {
        "fields": fields,
        "line_items": line_items,
        "raw": json_data,
        "model_output": sequence,
    }


def _map_donut_json_to_fields(data: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Map Donut token2json output into our canonical invoice fields.
    Supports Swedish OCR slips, EU invoices, and flat-list Donut outputs.
    """

    fields = {
        "invoice_no": None,
        "date": None,
        "subtotal": None,
        "tax": None,
        "total": None,
        "tax_rate": None,
    }
    line_items: List[Dict[str, Any]] = []

    if not data:
        return fields, line_items

    # Normalize root (Donut list → pseudo-menu)
    if isinstance(data, list):
        menu = data
    elif isinstance(data, dict) and "menu" in data:
        menu = data.get("menu")
    else:
        menu = []

    # Flatten all text content from menu items
    textbuf = json.dumps(menu, ensure_ascii=False)

    # ------------------------------
    # 1. Extract OCR/Fakturanummer
    # ------------------------------
    # Prefer 10+ digit numbers (OCR reference, Bankgiro)
    candidates = re.findall(r"\b(\d{8,12})\b", textbuf)
    if candidates:
        # If multiple, prefer 10-digit Swedish payment references
        tens = [c for c in candidates if len(c) == 10]
        fields["invoice_no"] = tens[0] if tens else candidates[0]

    # ------------------------------
    # 2. Extract total amount
    # ------------------------------
    # Scan for amounts like 172.0okr or 172,00 kr or 172.00
    amt = re.findall(r"(\d+[.,]\d{1,2})\s*(?:kr|okr|sek)?", textbuf, flags=re.IGNORECASE)
    if amt:
        fields["total"] = amt[-1].replace(",", ".")

    # ------------------------------
    # 3. Extract invoice/due date
    # ------------------------------
    date = re.search(r"(20\d{2}-\d{2}-\d{2})", textbuf)
    if date:
        fields["date"] = date.group(1)

    # ------------------------------
    # 4. Extract line items (very loose, Donut is weak here)
    # ------------------------------
    for item in menu:
        if not isinstance(item, dict):
            continue

        li = {
            "description": item.get("nm") or None,
            "quantity": item.get("cnt") or None,
            "unit_price": item.get("unitprice") or None,
            "line_total": item.get("price") if isinstance(item.get("price"), str) else None,
        }
        if any(li.values()):
            line_items.append(li)

    return fields, line_items

