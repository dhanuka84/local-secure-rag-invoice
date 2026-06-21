import os
import json
import math
import re
import hashlib
import tempfile
from typing import Any, Dict, Optional, List

from src.invoice.pdf_io import pdf_to_page_bundle, pdf_to_text_and_images, ocr_if_needed
from src.invoice.signature import build_signature
from src.invoice.template_cache import TemplateCache
from src.invoice.extract import extract_fields
from src.invoice.field_formats import normalize_date_string, normalize_field_value, parse_amount_string
from src.invoice.template_learner import learn_regexes
from src.invoice.vision_validate import validate_with_vision
from src.invoice.cerbos_client import can_promote_template
from src.invoice.metrics import TemplateMetrics
from src.invoice.layoutlm_extract import extract_with_layoutlm
from src.invoice.spacy_extract import extract_with_spacy
from src.invoice.ollama_runtime import EMBED_MODEL, TEXT_MODEL

from langchain_ollama.embeddings import OllamaEmbeddings
from src.invoice.template_learner import refine_regexes
from pymilvus import MilvusClient, DataType

DEFAULT_MILVUS_DIR = os.path.join(tempfile.gettempdir(), "local-secure-rag-invoice", "milvus")
os.makedirs(DEFAULT_MILVUS_DIR, exist_ok=True)
MILVUS_URI = os.getenv("INVOICE_MILVUS_URI", os.path.join(DEFAULT_MILVUS_DIR, "milvus_lite.db"))

COLL = "invoice_templates"

def _normalize_number(value: str | None) -> float | None:
    if not value:
        return None
    import re
    cleaned = re.sub(r"[^0-9.,-]", "", value)
    if not cleaned:
        return None
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def node_hybrid_extract_fields(state: dict) -> dict:
    """
    Combine Donut (doc_vlm) + LayoutLMv3 extraction and compute a validation score.
    Assumes doc_vlm already ran and filled state["fields"] and/or state["ml_line_items"].
    """
    pdf_path = state.get("pdf_path") or state.get("pdf")

    doc_fields = state.get("fields") or {}
    doc_items = state.get("ml_line_items") or []

    lm_result = extract_with_layoutlm(pdf_path)
    lm_fields = lm_result.get("fields") or {}
    lm_items = lm_result.get("line_items") or []

    # --- Merge fields (simple voting/priority rule) ---
    merged = dict(doc_fields)  # start from Donut

    def choose(key: str) -> str | None:
        a = (doc_fields or {}).get(key)
        b = (lm_fields or {}).get(key)
        # Prefer exact agreement
        if a and b and str(a) == str(b):
            return a
        # Otherwise prefer LayoutLM for invoice_no/date,
        # and Donut for totals (tune as you wish)
        if key in ("invoice_no", "date"):
            return b or a
        if key in ("total", "subtotal", "tax", "tax_rate"):
            return a or b
        return a or b

    for key in ["invoice_no", "date", "subtotal", "tax", "total", "tax_rate"]:
        merged[key] = choose(key)

    # --- Numeric validation: subtotal + tax ≈ total ---
    total = _normalize_number(merged.get("total"))
    subtotal = _normalize_number(merged.get("subtotal"))
    tax = _normalize_number(merged.get("tax"))

    math_ok = None
    if subtotal is not None and tax is not None and total is not None:
        if abs((subtotal + tax) - total) <= 0.01 * max(total, 1.0):
            math_ok = True
        else:
            math_ok = False

    # --- Agreement checks between models ---
    agree_total = None
    if doc_fields.get("total") and lm_fields.get("total"):
        agree_total = (
            _normalize_number(doc_fields["total"]) ==
            _normalize_number(lm_fields["total"])
        )

    agree_date = None
    if doc_fields.get("date") and lm_fields.get("date"):
        agree_date = (str(doc_fields["date"]) == str(lm_fields["date"]))

    agree_invoice = None
    if doc_fields.get("invoice_no") and lm_fields.get("invoice_no"):
        agree_invoice = (str(doc_fields["invoice_no"]) == str(lm_fields["invoice_no"]))

    # --- Build an ML validation score [0,1] ---
    score = 0.0
    components = 0

    if agree_total is not None:
        components += 1
        if agree_total:
            score += 0.3

    if agree_date is not None:
        components += 1
        if agree_date:
            score += 0.2

    if agree_invoice is not None:
        components += 1
        if agree_invoice:
            score += 0.2

    if math_ok is not None:
        components += 1
        if math_ok:
            score += 0.3

    if components > 0:
        score = min(1.0, score)  # already normalized by weights
    else:
        score = 0.0

    # --- Write back to state ---
    state["fields"] = merged
    # Combine line items (just concatenate for now)
    state["ml_line_items"] = (doc_items or []) + (lm_items or [])
    state["layoutlm_raw"] = lm_result.get("raw_tokens")
    state["layoutlm_device"] = lm_result.get("device")
    state["ml_validation_score"] = score

    return state


import threading

_milvus_client_instance = None
_milvus_client_lock = threading.Lock()

def _get_milvus_client() -> MilvusClient:
    global _milvus_client_instance
    print(f"[DEBUG] _get_milvus_client called. Current instance: {_milvus_client_instance}")
    with _milvus_client_lock:
        if _milvus_client_instance is None:
            print(f"[DEBUG] Creating new MilvusClient...")
            _milvus_client_instance = MilvusClient(uri=MILVUS_URI)
            print(f"[DEBUG] MilvusClient created successfully.")
        return _milvus_client_instance

def _milvus_connect():
    pass

def _milvus_ensure(dim: int = 768):
    client = _get_milvus_client()
    if not client.has_collection(COLL):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="signature", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="vendor", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=dim)
        
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec", index_type="HNSW", metric_type="IP", params={"M": 16, "efConstruction": 200})
        
        client.create_collection(
            collection_name=COLL,
            schema=schema,
            index_params=index_params
        )
    client.load_collection(COLL)

def _milvus_embed(signature: str, vendor: str) -> List[float]:
    text = f"template signature {signature} vendor {vendor}"
    try:
        emb = OllamaEmbeddings(model=EMBED_MODEL)
        return emb.embed_query(text)
    except Exception as exc:
        print(f"[milvus] Ollama embeddings unavailable; using deterministic local fallback. {exc}")
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: List[float] = []
        for i in range(768):
            byte = digest[i % len(digest)]
            values.append((byte / 255.0) * 2.0 - 1.0)
        return values

def _milvus_upsert(signature: str, vendor: str):
    client = _get_milvus_client()
    vec = _milvus_embed(signature, vendor)
    data = [
        {"signature": signature, "vendor": vendor, "vec": vec}
    ]
    client.upsert(collection_name=COLL, data=data)

def _milvus_suggest(signature: str, vendor: str, top_k: int = 3) -> List[str]:
    client = _get_milvus_client()
    qvec = _milvus_embed(signature, vendor)
    res = client.search(
        collection_name=COLL,
        data=[qvec],
        limit=top_k,
        output_fields=["signature"],
        search_params={"metric_type": "IP", "params": {"ef": 64}}
    )
    out = []
    for hits in res:
        for hit in hits:
            out.append(hit.get("entity", {}).get("signature"))
    return out

def node_extract_pdf(state):
    bundle = pdf_to_page_bundle(state["pdf_path"])
    state["text"] = bundle["text"]
    state["images"] = bundle["images"]
    state["page_texts"] = bundle["page_texts"]
    return state

def node_ocr_if_needed(state):
    state["text"] = ocr_if_needed(state["text"], state.get("images", []))
    return state


def node_structure_detect(state: dict) -> dict:
    page_sections = []
    header_keys = ["invoice", "faktura", "fakturadatum", "date", "ocr", "kundnummer"]
    recipient_keys = ["kund", "customer", "adress", "address", "till", "to"]
    summary_keys = ["summa", "subtotal", "moms", "tax", "total", "totalt belopp", "att betala"]
    payment_keys = ["ocr", "bankgiro", "autogiro", "iban", "swift", "betalning", "payment"]

    for page_index, page_text in enumerate(state.get("page_texts") or []):
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        sections = {
            "header": [],
            "recipient": [],
            "summary": [],
            "payment": [],
            "table": [],
            "other": [],
        }
        total_lines = max(len(lines), 1)

        for idx, line in enumerate(lines):
            lower = line.lower()
            line_position = idx / total_lines

            if any(key in lower for key in header_keys) and line_position <= 0.35:
                sections["header"].append(line)
            elif any(key in lower for key in recipient_keys):
                sections["recipient"].append(line)
            elif any(key in lower for key in summary_keys) or line_position >= 0.65 and re.search(r"\d[\d\s.,]*\s*(kr|%|$)", line, re.IGNORECASE):
                sections["summary"].append(line)
            elif any(key in lower for key in payment_keys):
                sections["payment"].append(line)
            elif re.search(r"\d", line) and len(line.split()) >= 4:
                sections["table"].append(line)
            else:
                sections["other"].append(line)

        page_sections.append(
            {
                "page_number": page_index + 1,
                "section_line_counts": {name: len(values) for name, values in sections.items()},
                "sections": sections,
            }
        )

    state["page_sections"] = page_sections
    return state

def node_signature(state):
    sig = build_signature(state["text"])
    state["signature"] = sig
    vendor = next((l.strip() for l in state["text"].splitlines() if l.strip()), "unknown")
    state["vendor"] = vendor[:64]
    return state

def node_check_cache(state):
    cache = TemplateCache()
    active = cache.get_active(state["signature"])
    staging = cache.get_staging(state["signature"])
    state["template_active"] = active
    state["template_staging"] = staging
    if active:
        state["template"] = active; state["template_source"] = "active"; state["template_resolution_mode"] = "cache_active"
    elif staging:
        state["template"] = staging; state["template_source"] = "staging"; state["template_resolution_mode"] = "cache_staging"
    else:
        state["template"] = None; state["template_source"] = "none"; state["template_resolution_mode"] = "cache_miss"
    return state

def should_reuse_or_search(state: dict) -> str:
    """
    Decide whether to reuse a template or run the search/learn path.

    We ONLY reuse templates that are already ACTIVE.
    Staging/learned templates still go through the "search" path,
    which can hit Donut + learning again.
    """
    template_source = state.get("template_source")

    if template_source == "active":
        return "reuse"
    else:
        # "staging", "learned", None, etc.
        return "search"


def node_milvus_suggest(state):
    sig, vendor = state["signature"], state["vendor"]
    try:
        _milvus_connect()
        _milvus_ensure()
        _milvus_upsert(sig, vendor)
        suggestions = _milvus_suggest(sig, vendor, top_k=3)
        state["embedding_backend"] = "ollama"
        state["embedding_model"] = EMBED_MODEL
    except Exception as exc:
        print(f"[milvus] Suggestion path unavailable; continuing without similar-template search. {exc}")
        suggestions = []
        state["embedding_backend"] = "local_hash_fallback"
        state["embedding_model"] = EMBED_MODEL

    state["suggested_signatures"] = suggestions

    cache = TemplateCache()
    for s in suggestions:
        t = cache.get_active(s) or cache.get_staging(s)
        if t:
            state["template"] = t
            state["template_source"] = "suggested"
            state["template_resolution_mode"] = "milvus_suggested"
            break

    return state

def should_use_suggest_or_learn(state) -> str:
    if state.get("template"):
        return "suggested"
    return "learn"

def node_learn_and_stage(state):
    tmpl = learn_regexes(state["text"])
    cache = TemplateCache()
    cache.set_staging(state["signature"], tmpl)
    state["template"] = tmpl
    state["template_source"] = "learned"
    state["template_resolution_mode"] = "learned_new"
    state["template_learning_backend"] = tmpl.get("_learning_backend", "unknown")
    state["template_learning_model"] = TEXT_MODEL
    return state


def _recover_missing_fields(fields: dict, page_sections: List[Dict[str, Any]]) -> dict:
    recovered = dict(fields)
    amount_token = r"([0-9]+(?:[ .][0-9]{3})*(?:[.,][0-9]{2})|[0-9]+(?:[.,][0-9]{2}))"

    section_lines: List[str] = []
    for page in page_sections or []:
        sections = page.get("sections") or {}
        for section_name in ("header", "summary", "payment", "table", "recipient", "other"):
            section_lines.extend(sections.get(section_name) or [])

    if not recovered.get("date"):
        for line in section_lines:
            m = re.search(r"\b(?:datum|date|fakturadatum)\b[:\s]*([0-9]{4}[-/.][0-9]{2}[-/.][0-9]{2}|[0-9]{2}[-/.][0-9]{2}[-/.][0-9]{4})", line, re.IGNORECASE)
            if m:
                recovered["date"] = normalize_date_string(m.group(1))
                break

    if not recovered.get("total"):
        total_patterns = [
            rf"att betala(?:\s+i\s+\w+)?[:\s]*{amount_token}",
            rf"totalt?\s+belopp[:\s]*{amount_token}",
            rf"summa att betala[:\s]*{amount_token}",
        ]
        for line in section_lines:
            for pattern in total_patterns:
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    amount = parse_amount_string(m.group(1))
                    if amount is not None:
                        recovered["total"] = str(amount)
                        break
            if recovered.get("total"):
                break

    if not recovered.get("subtotal"):
        # Prefer explicit subtotal-like lines.
        subtotal_patterns = [
            rf"summa\s+exkl\s+moms[:\s]*{amount_token}",
            rf"subtotal[:\s]*{amount_token}",
            rf"0%\s+{amount_token}$",
        ]
        for line in section_lines:
            for pattern in subtotal_patterns:
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    amount = parse_amount_string(m.group(1))
                    if amount is not None:
                        recovered["subtotal"] = str(amount)
                        break
            if recovered.get("subtotal"):
                break

        # Fallback: last amount on a likely line-item row.
        if not recovered.get("subtotal"):
            for line in section_lines:
                if any(keyword in line.lower() for keyword in ("avgift", "belopp", "period")):
                    amounts = re.findall(r"[0-9][0-9\s.,]*", line)
                    parsed = [parse_amount_string(a) for a in amounts]
                    parsed = [a for a in parsed if a is not None]
                    if parsed:
                        recovered["subtotal"] = str(parsed[-1])
                        break

    if not recovered.get("tax_rate") and recovered.get("tax"):
        tax_value = parse_amount_string(recovered.get("tax"))
        if tax_value == Decimal("0"):
            recovered["tax_rate"] = "0"

    return recovered


def node_extract_fields(state):
    from src.invoice.extract import extract_fields
    from decimal import Decimal
    template_fields = extract_fields(state["text"], state.get("template") or {})
    baseline_fields = extract_fields(state["text"], {"regex": {}})

    def _to_dec(x):
        if x is None:
            return None
        return Decimal(str(x))

    def _score(candidate: dict) -> tuple[int, int]:
        present = sum(1 for key in ("invoice_no", "date", "subtotal", "tax", "total") if candidate.get(key))
        subtotal = _to_dec(candidate.get("subtotal"))
        tax = _to_dec(candidate.get("tax"))
        total = _to_dec(candidate.get("total"))
        math_ok = 0
        if subtotal is not None and tax is not None and total is not None:
            math_ok = 1 if (total - (subtotal + tax)).copy_abs() <= Decimal("0.01") else 0
        return (math_ok, present)

    if _score(baseline_fields) > _score(template_fields):
        fields = baseline_fields
        state["field_extraction_backend"] = "bilingual_fallback"
    else:
        fields = template_fields
        state["field_extraction_backend"] = "template_or_default_regex"

    recovered_fields = _recover_missing_fields(fields, state.get("page_sections") or [])
    if _score(recovered_fields) >= _score(fields):
        fields = recovered_fields
        state["field_extraction_backend"] = "structure_recovery"

    state["fields"] = fields

    sub = fields.get("subtotal")
    tax = fields.get("tax")
    tot = fields.get("total")

    sd, td, Td = _to_dec(sub), _to_dec(tax), _to_dec(tot)

    consistent = True
    expected_total = None
    if sd is not None and td is not None:
        expected_total = sd + td
        if Td is not None and (Td - expected_total).copy_abs() > _to_dec("0.01"):
            consistent = False

    state["fields_consistent"] = consistent
    state["expected_total"] = str(expected_total) if expected_total is not None else None
    return state



import re
from decimal import Decimal, InvalidOperation

def _to_swedish_amount(value) -> str | None:
    """
    Convert dot-decimal numbers (172.00) to Swedish comma-decimal format (172,00 kr)
    for the vision model.
    """
    if value is None:
        return None
    s = str(value).strip()
    if re.match(r"^-?\d+(\.\d+)?$", s):
        s = s.replace(".", ",")
    return f"{s} kr"


def _build_validation_evidence(
    text: str,
    fields: dict,
    page_texts: Optional[List[str]] = None,
    page_sections: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lowered_keywords = [
        "invoice", "date", "subtotal", "tax", "total",
        "faktura", "fakturadatum", "summa", "moms", "belopp", "ocr",
    ]

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    def _build_page_section(page_number: int, raw_text: str) -> dict:
        page_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        keyword_lines = [line for line in page_lines if any(keyword in line.lower() for keyword in lowered_keywords)]

        candidates = []
        for line in keyword_lines:
            lowered = line.lower()
            tags = []
            for tag, keys in {
                "invoice_no": ["invoice", "ocr", "faktura"],
                "date": ["date", "fakturadatum"],
                "subtotal": ["subtotal", "summa exkl moms"],
                "tax": ["tax", "moms"],
                "total": ["total", "totalt belopp", "att betala", "dras via autogiro"],
            }.items():
                if any(key in lowered for key in keys):
                    tags.append(tag)
            candidates.append({"line": line, "tags": tags})

        field_evidence = {}
        for field_name, value in fields.items():
            if value is None:
                continue
            normalized_field_value = normalize_field_value(field_name, str(value)) or str(value)
            normalized_value = _norm(normalized_field_value)
            matches = []
            for entry in candidates:
                line = entry["line"]
                line_normalized = line
                if field_name == "date":
                    date_match = re.search(r"(\d{4}[-/.]\d{2}[-/.]\d{2}|\d{2}[-/.]\d{2}[-/.]\d{4})", line)
                    if date_match:
                        line_normalized = normalize_date_string(date_match.group(1)) or line
                if normalized_value and normalized_value in _norm(line_normalized):
                    matches.append({
                        "line": line,
                        "tags": entry["tags"],
                        "match_type": "direct",
                        "normalized_value": normalized_field_value,
                    })
                elif field_name in entry["tags"]:
                    matches.append({
                        "line": line,
                        "tags": entry["tags"],
                        "match_type": "tag",
                        "normalized_value": normalized_field_value,
                    })
            field_evidence[field_name] = matches[:5]

        return {
            "page_number": page_number,
            "top_lines": page_lines[:12],
            "candidate_lines": candidates[:30],
            "field_evidence": field_evidence,
            "sections": next((entry.get("sections") for entry in (page_sections or []) if entry.get("page_number") == page_number), {}),
            "line_count": len(page_lines),
        }

    page_sections = []
    for index, page_text in enumerate(page_texts or [text]):
        page_sections.append(_build_page_section(index + 1, page_text))

    field_evidence = {}
    for section in page_sections:
        for field_name, matches in section["field_evidence"].items():
            field_evidence.setdefault(field_name, [])
            for match in matches:
                annotated = dict(match)
                annotated["page_number"] = section["page_number"]
                field_evidence[field_name].append(annotated)

    # Add derived support for tax rate if subtotal and tax are present.
    subtotal = fields.get("subtotal")
    tax = fields.get("tax")
    tax_rate = fields.get("tax_rate")
    if subtotal and tax and tax_rate:
        try:
            sub_val = Decimal(str(subtotal).replace(",", "."))
            tax_val = Decimal(str(tax).replace(",", "."))
            rate_val = Decimal(str(tax_rate).replace(",", "."))
            derived_rate = (tax_val / sub_val).quantize(Decimal("0.0001")) if sub_val != 0 else None
            if derived_rate is not None:
                field_evidence.setdefault("tax_rate", [])
                field_evidence["tax_rate"].append(
                    {
                        "line": f"derived from subtotal={subtotal} and tax={tax}",
                        "tags": ["tax_rate"],
                        "match_type": "derived",
                        "derived_rate": str(derived_rate),
                        "expected_rate": str(rate_val),
                    }
                )
        except Exception:
            pass

    return {
        "top_lines": lines[:12],
        "pages": page_sections,
        "candidate_lines": [item for section in page_sections for item in section["candidate_lines"]][:40],
        "field_evidence": field_evidence,
        "extracted_fields": fields,
        "normalized_fields": {
            key: normalize_field_value(key, str(value))
            for key, value in fields.items()
            if value is not None
        },
        "line_count": len(lines),
    }


def node_vision_validate(state: dict) -> dict:
    """
    Hybrid vision validation:
      1. Call the vision model with Swedish-formatted values.
      2. If our extracted fields are complete and math-consistent,
         override the model and mark this as a strong PASS.
    """
    from src.invoice.vision_validate import validate_with_vision

    # WARNING: at this stage, fields still use ENGLISH keys
    # (invoice_no, date, subtotal, tax, total, tax_rate).
    # They are only renamed to Swedish in node_done.
    fields = state.get("fields") or {}

    # --- 1) Prepare payload for vision model (Swedish formatting) ---
    vis_fields = {}
    for k, v in fields.items():
        if k in ("subtotal", "tax", "total"):
            vis_fields[k] = _to_swedish_amount(v)
        elif k == "tax_rate":
            # convert 0.25 -> "25%"
            try:
                rate = float(str(v).replace(",", "."))
                vis_fields[k] = f"{int(rate * 100)}%"
            except Exception:
                vis_fields[k] = str(v) if v is not None else None
        else:
            vis_fields[k] = str(v) if v is not None else None

    evidence = _build_validation_evidence(
        state.get("text", ""),
        vis_fields,
        state.get("page_texts"),
        state.get("page_sections"),
    )
    verdict = validate_with_vision(vis_fields, state.get("images", []), evidence=evidence)
    strict_vision_required = state.get("template_source") != "active"
    state["validation_mode"] = "strict" if strict_vision_required else "audit"

    vpass = bool(verdict.get("pass"))
    vscore = float(verdict.get("score", 0.0))
    vcrit = verdict.get("critique", "")

    # --- 2) Deterministic override based on extracted fields ---
    def _dec(x):
        if x is None:
            return None
        try:
            return Decimal(str(x))
        except (InvalidOperation, ValueError):
            return None

    core_ok = all(fields.get(k) for k in ("invoice_no", "date", "subtotal", "tax", "total"))
    missing_core_fields = [k for k in ("invoice_no", "date", "subtotal", "tax", "total") if not fields.get(k)]

    sub = _dec(fields.get("subtotal"))
    tax = _dec(fields.get("tax"))
    tot = _dec(fields.get("total"))

    math_ok = False
    if sub is not None and tax is not None and tot is not None:
        diff = (sub + tax) - tot
        # allow some öre rounding tolerance
        math_ok = diff.copy_abs() <= Decimal("0.50")

    if core_ok and math_ok and not strict_vision_required:
        # We trust our extraction more than the raw VLM score.
        vpass = True
        if vscore < 0.9:
            vscore = 0.95
        if not vcrit:
            vcrit = "Fields are complete and subtotal + moms ≈ totalt belopp."

    state["vision_required"] = strict_vision_required
    state["audit_warning"] = False
    state["audit_reason"] = ""
    if not strict_vision_required:
        if verdict.get("backend") in {"timeout", "unavailable", "ollama_unparsed"}:
            state["audit_warning"] = True
            state["audit_reason"] = f"Audit validation backend issue: {verdict.get('backend')}"
        elif not bool(verdict.get("pass")):
            state["audit_warning"] = True
            state["audit_reason"] = vcrit or "Audit validation disagreed with active template extraction."

    if strict_vision_required:
        backend_ok = verdict.get("backend") in {"ollama", "ollama_json", "ollama_multimodal"}
        if core_ok and math_ok and backend_ok and vpass:
            state["validation_decision"] = "pass"
            state["combined_validation_score"] = max(vscore, 0.9)
        else:
            vpass = False
            vscore = 0.0
            reasons = []
            if missing_core_fields:
                reasons.append(f"missing required fields: {', '.join(missing_core_fields)}")
            if not backend_ok:
                reasons.append(f"validation backend not accepted: {verdict.get('backend')}")
            if core_ok and math_ok not in (None, True):
                reasons.append("math consistency check failed")
            if not reasons and vcrit:
                reasons.append(vcrit)
            vcrit = "Strict validation failed: " + "; ".join(reasons or ["validation did not pass"])
            state["validation_decision"] = "review"
            state["combined_validation_score"] = 0.0
    else:
        if core_ok and math_ok:
            state["validation_decision"] = "pass"
            state["combined_validation_score"] = max(vscore, 0.95 if vpass else 1.0)
            vpass = True
        else:
            vpass = False
            state["validation_decision"] = "review"
            state["combined_validation_score"] = 0.0

    state["vision_pass"] = vpass
    state["vision_score"] = vscore
    state["vision_critique"] = vcrit
    state["vision_backend"] = verdict.get("backend", "unknown")
    state["vision_model"] = verdict.get("model")
    state["validation_evidence"] = evidence
    state["vision_fields_payload"] = vis_fields  # debug

    return state


def should_pass_or_review(state: dict) -> str:
    return state.get("validation_decision", "review")



def node_done(state):
    """
    Final node. Optionally records success metrics.
    """
    sig = state.get("signature")
    if sig:
        # Consider a run "successful" if:
        #   - vision_pass is True
        #   - we have fields
        vpass = state.get("vision_pass", False)
        fields = state.get("fields") or {}
        if vpass and fields:
            TemplateMetrics().record_success(sig)

    state["done"] = True
    return state


import os
from src.invoice.template_cache import TemplateCache
from src.invoice.metrics import TemplateMetrics

DOCVLM_DEBUG = os.getenv('DOCVLM_DEBUG', 'false').lower() == 'true'
from src.invoice.cerbos_client import can_promote_template

AUTO_PROMOTE_THRESHOLD = int(os.getenv("AUTO_PROMOTE_THRESHOLD", "1"))


def node_promote_template(state: dict) -> dict:
    sig = state.get("signature")
    stage = state.get("template_source", "staging")
    role = state.get("role") or os.getenv("APP_ROLE", "employee")

    cache = TemplateCache()
    metrics = TemplateMetrics()

    # 🔹 1) If already active, do NOT try to promote again
    if stage == "active":
        # Keep previous status if it was "promoted", otherwise mark as already_active
        state.setdefault("promotion_status", "already_active")
        state.setdefault("template_resolution_mode", "cache_active")
        return state

    # 🔹 2) Read and normalize success_count
    m = metrics.get(sig) if sig else {}
    raw_success = m.get("success_count", 0)
    try:
        success_count = int(raw_success)
    except (TypeError, ValueError):
        success_count = 0

    current_run_success = 1 if state.get("vision_pass") and (state.get("fields") or {}) else 0
    effective_success_count = success_count + current_run_success

    # 🔹 3) Respect AUTO_PROMOTE_THRESHOLD for non-active templates
    threshold = AUTO_PROMOTE_THRESHOLD
    if effective_success_count < threshold:
        state["promotion_status"] = f"pending_success_{effective_success_count}"
        return state

    # 🔹 4) Ask Cerbos if this role may promote this stage
    allowed = can_promote_template(role=role, stage=stage)
    if not allowed:
        state["promotion_status"] = "denied"
        return state

    # 🔹 5) Promote in cache
    if stage == "suggested":
        # Adopt a validated suggested template for the current signature.
        template = state.get("template")
        if template and sig:
            cache.set_active(sig, template)
            state["template_source"] = "active"
            state["template_resolution_mode"] = "promoted_from_suggested"
            state["promotion_status"] = "promoted_from_suggested"
            metrics.record_promotion(sig)
            return state
        state["promotion_status"] = "promote_failed"
        return state

    promoted = cache.promote(sig)
    if promoted:
        state["template_source"] = "active"
        state["template_resolution_mode"] = "promoted_from_staging"
        state["promotion_status"] = "promoted"
        metrics.record_promotion(sig)
    else:
        state["promotion_status"] = "promote_failed"

    return state



def _to_swedish_str(value) -> str | None:
    """
    Convert '172.00' -> '172,00' for final JSON fields.
    Does NOT add 'kr'.
    """
    if value is None:
        return None
    s = str(value).strip()
    # Only touch simple numeric formats
    if re.match(r"^-?\d+(\.\d+)?$", s):
        s = s.replace(".", ",")
    return s



def node_done(state: dict) -> dict:
    """
    Final node.
    - Records success metrics
    - Converts numeric fields to Swedish comma decimals WITH ' kr'
    - Converts tax_rate to '25%'
    - Renames output fields to Swedish labels
    """
    import re
    sig = state.get("signature")
    fields = state.get("fields") or {}

    # Success metric
    if sig:
        vpass = state.get("vision_pass", False)
        if vpass and fields:
            TemplateMetrics().record_success(sig)

    # Convert numeric money fields
    def _to_swedish_money(value):
        if value is None:
            return None
        s = str(value).strip()
        if re.match(r"^-?\d+(\.\d+)?$", s):
            s = s.replace(".", ",")
        return f"{s} kr"

    # Internal → Swedish formatting
    if fields.get("subtotal") is not None:
        fields["subtotal"] = _to_swedish_money(fields["subtotal"])
    if fields.get("tax") is not None:
        fields["tax"] = _to_swedish_money(fields["tax"])
    if fields.get("total") is not None:
        fields["total"] = _to_swedish_money(fields["total"])

    # Tax rate: convert 0.25 → "25%"
    if "tax_rate" in fields and fields["tax_rate"] is not None:
        try:
            rate = float(str(fields["tax_rate"]).replace(",", "."))
            fields["tax_rate"] = f"{int(rate * 100)}%"
        except Exception:
            fields["tax_rate"] = str(fields["tax_rate"])

    # Map English → Swedish keys
    swedish = {
        "invoice_no": "OCR-/fakturanummer",
        "date": "Fakturadatum",
        "subtotal": "Summa exkl moms",
        "tax": "Moms",
        "total": "Totalt belopp",
        "tax_rate": "Moms (%)"
    }

    swedish_fields = {}
    for eng_key, swe_key in swedish.items():
        if eng_key in fields:
            swedish_fields[swe_key] = fields[eng_key]

    # Replace state fields with Swedish keys
    state["fields"] = swedish_fields
    state["done"] = True
    return state


def node_mark_for_review(state):
    state["done"] = False
    return state


def node_auto_refine_template(state):
    """If fields are inconsistent or vision failed, try to refine the template once."""
    tmpl = state.get("template")
    if not tmpl:
        state["auto_refine_status"] = "no_template"
        return state

    attempts = int(state.get("refine_attempts", 0))
    if attempts >= 1:
        state["auto_refine_status"] = "skipped_max_attempts"
        return state

    fields = state.get("fields", {}) or {}
    consistent = bool(state.get("fields_consistent", True))
    vision_pass = bool(state.get("vision_pass", False))
    expected_total = state.get("expected_total")
    current_total = fields.get("total")

    if consistent and vision_pass:
        state["auto_refine_status"] = "skipped_no_error"
        state["refine_attempts"] = attempts + 1
        return state

    if not expected_total or not current_total or str(expected_total) == str(current_total):
        state["auto_refine_status"] = "skipped_no_total_hint"
        state["refine_attempts"] = attempts + 1
        return state

    new_tmpl = refine_regexes(
        invoice_text=state["text"],
        current_template=tmpl,
        field_name="total",
        expected=str(expected_total),
        got=str(current_total),
    )

    sig = state.get("signature")
    if sig:
        from src.invoice.template_cache import TemplateCache
        TemplateCache().set_staging(sig, new_tmpl)
        TemplateMetrics().record_refine(sig)

    state["template"] = new_tmpl
    state["template_source"] = "refined"
    state["refine_attempts"] = attempts + 1
    state["auto_refine_status"] = "refined_total"
    return state


def node_doc_vlm_extract_fields(state: dict) -> dict:
    pdf_path = state.get("pdf_path") or state.get("pdf")

    if DOCVLM_DEBUG:
        print("------ DOCVLM NODE CALLED ------")
        print("PDF:", pdf_path)

    if not pdf_path:
        state.setdefault("fields", {})
        state.setdefault("ml_line_items", [])
        state["template_source"] = "doc_vlm"
        return state

    result = extract_with_doc_vlm(pdf_path)

    if DOCVLM_DEBUG:
        print("------ RAW DONUT SEQUENCE ------")
        print(result.get("model_output"))
        print("------ DONUT JSON ------")
        try:
            print(json.dumps(result.get("raw"), ensure_ascii=False, indent=2))
        except Exception:
            print(result.get("raw"))

    fields = result.get("fields") or {}
    line_items = result.get("line_items") or []

    current = state.get("fields") or {}
    current.update({
        "invoice_no": fields.get("invoice_no", current.get("invoice_no")),
        "date": fields.get("date", current.get("date")),
        "subtotal": fields.get("subtotal", current.get("subtotal")),
        "tax": fields.get("tax", current.get("tax")),
        "total": fields.get("total", current.get("total")),
        "tax_rate": fields.get("tax_rate", current.get("tax_rate")),
    })
    state["fields"] = current
    state["ml_line_items"] = line_items

    state["doc_vlm_raw"] = result.get("raw")
    state["doc_vlm_output"] = result.get("model_output")
    state["template_source"] = "doc_vlm"
    return state
