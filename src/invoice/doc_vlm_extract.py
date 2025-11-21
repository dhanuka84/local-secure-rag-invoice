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
    fields: Dict[str, Any] = {
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

    # Handle both dict- and list-rooted token2json outputs
    if isinstance(data, dict):
        root = data
    elif isinstance(data, list):
        # Many Donut configs return a list of sections; often we just want the first item
        if len(data) == 1 and isinstance(data[0], dict):
            root = data[0]
        else:
            # Fallback: wrap the list into a pseudo-root
            root = {"menu": data}
    else:
        # Unknown structure, bail out
        return fields, line_items

    # CORD-style: {"menu": [...], "sub_total": {...}, "total": {...}}
    menu = root.get("menu") or []
    sub_total = root.get("sub_total") or {}
    total_info = root.get("total") or sub_total


    text_for_ids = json.dumps(menu, ensure_ascii=False)
    id_matches = re.findall(r"\b(\d{8,12})\b", text_for_ids)
    if id_matches:
        ten = [m for m in id_matches if len(m) == 10]
        fields["invoice_no"] = ten[0] if ten else id_matches[0]

    if isinstance(total_info, dict):
        price = (
            total_info.get("total_price")
            or total_info.get("subtotal_price")
            or total_info.get("creditcardprice")
            or None
        )
        if price:
            cleaned = re.sub(r"[^0-9.,]", "", str(price))
            cleaned = cleaned.replace(",", ".")
            fields["total"] = cleaned or None

    if isinstance(sub_total, dict):
        tax_rate_raw = sub_total.get("tax_price") or None
        if tax_rate_raw:
            m = re.search(r"([0-9]+(?:[.,][0-9]+)?)%", str(tax_rate_raw))
            if m:
                val = m.group(1).replace(",", ".")
                try:
                    fields["tax_rate"] = str(float(val) / 100.0)
                except ValueError:
                    pass

    dump_text = json.dumps(data, ensure_ascii=False)
    m_date = re.search(r"(20\d{2}-\d{2}-\d{2})", dump_text)
    if m_date:
        fields["date"] = m_date.group(1)

    for item in menu:
        if not isinstance(item, dict):
            continue
        desc = item.get("nm")
        qty = item.get("cnt")
        unit_price = item.get("unitprice")
        price = item.get("price")

        li = {
            "description": desc,
            "quantity": qty,
            "unit_price": unit_price,
            "line_total": price,
            "tax_rate": None,
        }
        if any(v is not None for v in li.values()):
            line_items.append(li)

    return fields, line_items
