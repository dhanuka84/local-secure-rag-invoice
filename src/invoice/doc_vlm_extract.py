
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from src.invoice.pdf_utils import pdf_to_images


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

def _decode_donut_output(sequence: str) -> Dict[str, Any]:
    start = sequence.find("{")
    end = sequence.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(sequence[start:end+1])
    except Exception:
        return {}

def extract_with_doc_vlm(pdf_path: str, page_index: int = 0) -> Dict[str, Any]:
    processor, model = _load_doc_vlm()
    device = _get_device()

    images = pdf_to_images(pdf_path)
    if not images:
        return {"fields": {}, "line_items": [], "raw": {}, "model_output": ""}

    if page_index < 0 or page_index >= len(images):
        page_index = 0

    img = images[page_index]
    if isinstance(img, str):
        image = Image.open(img).convert("RGB")
    else:
        image = img.convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=MAX_LENGTH,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )

    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    parsed = _decode_donut_output(sequence)
    fields, line_items = _map_parsed_to_fields(parsed)

    return {
        "fields": fields,
        "line_items": line_items,
        "raw": parsed,
        "model_output": sequence,
    }

def _map_parsed_to_fields(parsed: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fields = {
        "invoice_no": None,
        "date": None,
        "subtotal": None,
        "tax": None,
        "total": None,
        "tax_rate": None,
    }
    line_items: List[Dict[str, Any]] = []

    if not parsed:
        return fields, line_items

    fields["invoice_no"] = parsed.get("invoice_number") or parsed.get("inv_no") or parsed.get("invoice_no")
    fields["date"] = parsed.get("date") or parsed.get("issue_date") or parsed.get("invoice_date")
    fields["total"] = parsed.get("total") or parsed.get("total_amount") or parsed.get("grand_total")
    fields["subtotal"] = parsed.get("subtotal") or parsed.get("sub_total") or parsed.get("net_amount")
    fields["tax"] = parsed.get("tax") or parsed.get("vat_amount") or parsed.get("tax_amount")
    fields["tax_rate"] = parsed.get("tax_rate") or parsed.get("vat_rate") or parsed.get("tax_percentage")

    items = parsed.get("items") or parsed.get("line_items") or parsed.get("products") or []
    if isinstance(items, dict):
        items = items.get("item", [])

    for item in items:
        if not isinstance(item, dict):
            continue
        li = {
            "description": item.get("description") or item.get("item_name") or item.get("name"),
            "quantity": item.get("quantity") or item.get("qty"),
            "unit_price": item.get("unit_price") or item.get("price") or item.get("unitPrice"),
            "tax_rate": item.get("tax_rate") or item.get("vat_rate"),
            "line_total": item.get("total_price") or item.get("amount") or item.get("line_total"),
        }
        line_items.append(li)

    return fields, line_items
