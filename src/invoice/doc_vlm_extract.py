import os
import re
import json
from typing import Any, Dict, List, Optional

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image

DOCVLM_MODEL_ID = os.getenv(
    "DOCVLM_MODEL_ID", "naver-clova-ix/donut-base-finetuned-cord-v2"
)
DOCVLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOCVLM_DEBUG = os.getenv("DOCVLM_DEBUG", "false").lower() == "true"

_processor: Optional[DonutProcessor] = None
_model: Optional[VisionEncoderDecoderModel] = None


def _lazy_load_donut() -> None:
    global _processor, _model
    if _processor is not None and _model is not None:
        return

    _processor = DonutProcessor.from_pretrained(DOCVLM_MODEL_ID)
    _model = VisionEncoderDecoderModel.from_pretrained(DOCVLM_MODEL_ID)
    _model.to(DOCVLM_DEVICE)
    _model.eval()


# --- simple helpers ---------------------------------------------------------


def _normalize_amount(raw: str) -> Optional[str]:
    """
    Normalize a Swedish-style money string to a plain decimal like '172.00'.

    Handles:
      - '172,0 ktr'   -> '172.00'
      - '172,000 kra' -> '172.00'
      - '95,79其'     -> '95.79'

    Returns None if the value is clearly not a sane money amount.
    """
    if not raw:
        return None

    s = raw.strip()

    # Keep only digits, comma and dot
    cleaned = re.sub(r"[^\d,\.]", "", s)
    if not cleaned:
        return None

    # If both comma and dot exist, assume comma is thousands sep, dot is decimal (US style)
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")

    # If only comma exists, treat it as decimal separator (Swedish style)
    elif "," in cleaned:
        cleaned = cleaned.replace(".", "")  # just in case
        cleaned = cleaned.replace(",", ".")

    # Now we should have only digits and at most one dot
    if cleaned.count(".") > 1:
        return None

    try:
        val = float(cleaned)
    except Exception:
        return None

    # Sanity bounds for invoice totals (0 < amount < 1M)
    if not (0 < val < 1_000_000):
        return None

    return f"{val:.2f}"




def _parse_sequence_to_records(seq: str) -> List[Dict[str, Any]]:
    """
    Very simple XML-ish tag parser for the Donut output.

    We don't depend on a specific schema (CORD), just the <s_xxx> ... </s_xxx> tags and <sep/>.
    Returns a list of small dicts: [{"nm": "...", "price": "...", ...}, ...]
    """

    # Strip leading task token etc.
    # Keep tags like <s_nm>, <s_price>, <sep/>
    # We treat <sep/> as record boundary.
    records: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}

    # Split on tags while keeping them
    tokens = re.split(r"(<[^>]+>)", seq)
    current_field = None

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        # Tag?
        if tok.startswith("<") and tok.endswith(">"):
            # separator?
            if tok == "<sep/>":
                if current:
                    records.append(current)
                    current = {}
                    current_field = None
                continue

            # Closing tag?
            if tok.startswith("</s_"):
                current_field = None
                continue

            # Opening tag <s_xxx> or self-closing
            m = re.match(r"<s_([^>/]+)>", tok)
            if m:
                current_field = m.group(1)
                # Initialize nested value if not present
                if current_field not in current:
                    current[current_field] = ""
            continue

        # Plain text
        if current_field is not None:
            if isinstance(current[current_field], str):
                if current[current_field]:
                    current[current_field] += " " + tok
                else:
                    current[current_field] = tok

    if current:
        records.append(current)

    return records


def _records_to_fields(all_records: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """
    Aggregate over all pages' records to get invoice_no, date, total, subtotal, tax, tax_rate.

    Changes:
      - Only treat numbers as money if they have a currency marker nearby ('kr', 'kra', 'ktr', 'sek', 'kronor', etc.).
      - Prefer total candidates from payment-related lines.
      - Infer VAT from the 'moms'/'mors' line and derive subtotal + tax_rate.
    """

    invoice_no: Optional[str] = None
    date: Optional[str] = None
    total: Optional[str] = None
    tax: Optional[str] = None
    subtotal: Optional[str] = None
    tax_rate: Optional[str] = None

    def rec_text(r: Dict[str, Any]) -> str:
        return " ".join(str(v) for v in r.values() if isinstance(v, str))

    # ------------------------------------------------------------------ #
    # 1) Invoice number + date from any page                             #
    # ------------------------------------------------------------------ #
    for r in all_records:
        nm = str(r.get("nm", "") or r.get("name", ""))
        price = str(r.get("price", ""))
        num_field = str(r.get("num", ""))

        text = f"{nm} {price} {num_field}"

        # Date: yyyy-mm-dd
        if not date:
            m_date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
            if m_date:
                date = m_date.group(1)

        # Invoice / OCR number with explicit cue
        if ("OCR" in nm.upper() or "FAKTURA" in nm.upper()
                or "FAKTURANUMMER" in nm.upper()
                or "FAKTURAUNUMER" in nm.upper()):
            m_id = re.search(r"\b(\d{8,20})\b", text)
            if m_id:
                invoice_no = m_id.group(1)

    # Fallback: any long-ish numeric token as invoice_no
    if not invoice_no:
        for r in all_records:
            for v in r.values():
                if not isinstance(v, str):
                    continue
                m_id = re.search(r"\b(\d{8,20})\b", v)
                if m_id:
                    invoice_no = m_id.group(1)
                    break
            if invoice_no:
                break

    # ------------------------------------------------------------------ #
    # 2) Collect money candidates only where a currency marker is nearby #
    # ------------------------------------------------------------------ #
    amount_pattern = re.compile(r"\b\d[\d,\.]{0,9}\b")
    currency_pattern = re.compile(r"k\s*r|kra|krs|kronor|sek|ktr", re.IGNORECASE)

    total_keywords = [
        "autogiro",
        "inbetalning",
        "girering",
        "avi",
        "bankgiro",
        "bankfiro",
        "bankfironummer",
        "bankfironumner",  # OCR-ish
        "dras via",
        "att betala",
        "belopp att betala",
    ]

    candidates: List[tuple] = []  # (value_float, norm_str, is_payment_line)

    for r in all_records:
        nm = str(r.get("nm", "")).lower()
        texts = [
            str(r.get("price", "")),
            str(r.get("num", "")),
            nm,
        ]

        is_payment_line = any(kw in nm for kw in total_keywords)

        for t in texts:
            if not isinstance(t, str) or not t:
                continue

            for m in amount_pattern.finditer(t):
                raw = m.group(0)

                # Skip very long pure-digit tokens (likely IDs)
                only_digits = re.sub(r"\D", "", raw)
                if len(only_digits) >= 8:
                    continue

                # Require a currency marker near the amount
                start, end = m.span()
                ctx_start = max(0, start - 8)
                ctx_end = min(len(t), end + 8)
                ctx = t[ctx_start:ctx_end]

                if not currency_pattern.search(ctx):
                    # e.g. '3120 kWh', '5331-9901', plain kWh or IDs: ignore
                    continue

                norm = _normalize_amount(raw)
                if norm is None:
                    continue

                try:
                    v = float(norm)
                except Exception:
                    continue

                candidates.append((v, norm, is_payment_line))

    # ------------------------------------------------------------------ #
    # 3) Choose total: prefer payment-related lines                      #
    # ------------------------------------------------------------------ #
    total_val: Optional[float] = None

    payment_candidates = [c for c in candidates if c[2]]
    if payment_candidates:
        v, s, _ = max(payment_candidates, key=lambda x: x[0])
        total_val, total = v, s
    elif candidates:
        v, s, _ = max(candidates, key=lambda x: x[0])
        total_val, total = v, s

    # ------------------------------------------------------------------ #
    # 4) Infer VAT ('moms' / 'mors') + subtotal + tax_rate               #
    # ------------------------------------------------------------------ #
    if total_val is not None:
        moms_amounts: List[float] = []
        moms_strs: List[str] = []

        for r in all_records:
            t = rec_text(r)
            t_lower = t.lower()
            if "moms" in t_lower or "mors" in t_lower:
                for m in amount_pattern.finditer(t):
                    raw = m.group(0)

                    # Also require currency context here
                    start, end = m.span()
                    ctx_start = max(0, start - 8)
                    ctx_end = min(len(t), end + 8)
                    ctx = t[ctx_start:ctx_end]

                    if not currency_pattern.search(ctx):
                        continue

                    norm = _normalize_amount(raw)
                    if not norm:
                        continue
                    try:
                        v = float(norm)
                    except Exception:
                        continue
                    # VAT should be < total
                    if 0 < v < total_val:
                        moms_amounts.append(v)
                        moms_strs.append(norm)

        if moms_amounts:
            # VAT is usually the largest amount < total among these
            idx = moms_amounts.index(max(moms_amounts))
            tax_val = moms_amounts[idx]
            tax = moms_strs[idx]

            sub_val = total_val - tax_val
            if sub_val > 0:
                subtotal = f"{sub_val:.2f}"

            # Approximate tax rate
            try:
                if subtotal:
                    sub_f = float(subtotal)
                    rate = tax_val / sub_f
                    if 0.05 < rate < 0.35:  # 5%–35% plausible VAT
                        tax_rate = f"{rate:.4f}"
            except Exception:
                pass

    return {
        "invoice_no": invoice_no,
        "date": date,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "tax_rate": tax_rate,
    }





# --- main API ---------------------------------------------------------------


def extract_with_doc_vlm(pdf_path: str) -> Dict[str, Any]:
    """
    Multi-page Donut extraction.

    - Renders every page to an image.
    - Runs Donut on each page.
    - Parses each page's sequence to JSON-like records.
    - Aggregates all records across pages into final fields.
    """
    _lazy_load_donut()
    assert _processor is not None
    assert _model is not None

    if DOCVLM_DEBUG:
        print("\n------ DOCVLM NODE CALLED ------")
        print(f"PDF: {pdf_path}")

    images: List[Image.Image] = convert_from_path(pdf_path, dpi=150)
    all_records: List[Dict[str, Any]] = []
    page_json: List[List[Dict[str, Any]]] = []

    task_prompt = "<s_cord-v2>"

    for page_idx, img in enumerate(images):
        # 1) Prepare inputs
        decoder_input_ids = _processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        pixel_values = _processor(img, return_tensors="pt").pixel_values

        pixel_values = pixel_values.to(DOCVLM_DEVICE)
        decoder_input_ids = decoder_input_ids.to(DOCVLM_DEVICE)

        # 2) Generate sequence
        outputs = _model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=_model.decoder.config.max_position_embeddings,
            pad_token_id=_processor.tokenizer.pad_token_id,
            eos_token_id=_processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[ _processor.tokenizer.unk_token_id ]],
        )

        sequence = _processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Often the first task token (<s_cord-v2>) remains; strip generic <...> at start
        sequence = re.sub(r"^<[^>]+>", "", sequence).strip()

        if DOCVLM_DEBUG:
            print(f"\n------ RAW DONUT SEQUENCE (page {page_idx+1}) ------")
            print(sequence)

        # 3) Parse into records
        records = _parse_sequence_to_records(sequence)
        page_json.append(records)
        all_records.extend(records)

    if DOCVLM_DEBUG:
        print("\n------ DONUT JSON (ALL PAGES) ------")
        print(json.dumps(page_json, indent=2, ensure_ascii=False))

    fields = _records_to_fields(all_records)

    return {
        "fields": fields,
        "line_items": [],     # you can later populate from all_records
        "raw_json": page_json,
    }
