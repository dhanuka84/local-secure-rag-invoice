import os
import json


from src.invoice.pdf_io import pdf_to_text_and_images, ocr_if_needed
from src.invoice.signature import build_signature
from src.invoice.template_cache import TemplateCache
from src.invoice.extract import extract_fields
from src.invoice.template_learner import learn_regexes
from src.invoice.vision_validate import validate_with_vision
from src.invoice.cerbos_client import can_promote_template
from src.invoice.metrics import TemplateMetrics
from src.invoice.doc_vlm_extract import extract_with_doc_vlm

DOCVLM_DEBUG = os.getenv('DOCVLM_DEBUG', 'false').lower() == 'true'
from src.invoice.template_learner import refine_regexes

from typing import List
from langchain_ollama.embeddings import OllamaEmbeddings
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

COLL = "invoice_templates"

def _milvus_connect():
    connections.connect("default", host="localhost", port="19530")

def _milvus_ensure(dim: int = 768):
    if not utility.has_collection(COLL):
        fields = [
            FieldSchema(name="signature", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="vendor", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, "invoice template embeddings")
        Collection(COLL, schema, shards_num=1)
    c = Collection(COLL)
    if not any(i.field_name == "vec" for i in c.indexes):
        c.create_index(field_name="vec", index_params={"index_type":"HNSW","metric_type":"IP","params":{"M":16,"efConstruction":200}})
    c.load()

def _milvus_embed(signature: str, vendor: str) -> List[float]:
    emb = OllamaEmbeddings(model="nomic-embed-text")
    return emb.embed_query(f"template signature {signature} vendor {vendor}")

def _milvus_upsert(signature: str, vendor: str):
    c = Collection(COLL)
    vec = _milvus_embed(signature, vendor)
    c.insert([[signature],[vendor],[vec]])
    c.flush(); c.load()

def _milvus_suggest(signature: str, vendor: str, top_k: int = 3) -> List[str]:
    c = Collection(COLL)
    qvec = [_milvus_embed(signature, vendor)]
    res = c.search(data=qvec, anns_field="vec", param={"metric_type":"IP","params":{"ef":64}},
                   limit=top_k, output_fields=["signature"])
    out = []
    for hit in res[0]:
        out.append(hit.entity.get("signature"))
    return out

def node_extract_pdf(state):
    text, images = pdf_to_text_and_images(state["pdf_path"])
    state["text"], state["images"] = text, images
    return state

def node_ocr_if_needed(state):
    state["text"] = ocr_if_needed(state["text"], state.get("images", []))
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
        state["template"] = active; state["template_source"] = "active"
    elif staging:
        state["template"] = staging; state["template_source"] = "staging"
    else:
        state["template"] = None; state["template_source"] = "none"
    return state

def should_reuse_or_search(state) -> str:
    return "reuse" if state.get("template") else "search"

def node_milvus_suggest(state):
    _milvus_connect(); _milvus_ensure()
    sig, vendor = state["signature"], state["vendor"]
    _milvus_upsert(sig, vendor)
    state["suggested_signatures"] = _milvus_suggest(sig, vendor, top_k=3)
    return state

def should_use_suggest_or_learn(state) -> str:
    cache = TemplateCache()
    for s in state.get("suggested_signatures", []):
        t = cache.get_active(s) or cache.get_staging(s)
        if t:
            state["template"] = t
            state["template_source"] = "suggested"
            return "suggested"
    return "learn"

def node_learn_and_stage(state):
    tmpl = learn_regexes(state["text"])
    cache = TemplateCache()
    cache.set_staging(state["signature"], tmpl)
    state["template"] = tmpl
    state["template_source"] = "learned"
    return state


def node_extract_fields(state):
    from src.invoice.extract import extract_fields
    fields = extract_fields(state["text"], state["template"])
    state["fields"] = fields

    from decimal import Decimal

    sub = fields.get("subtotal")
    tax = fields.get("tax")
    tot = fields.get("total")

    def _to_dec(x):
        if x is None:
            return None
        return Decimal(str(x))

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



def node_vision_validate(state):
    from src.invoice.vision_validate import validate_with_vision
    verdict = validate_with_vision(
        {k: (str(v) if v is not None else None) for k, v in state.get("fields", {}).items()},
        state.get("images", [])
    )
    state["vision_pass"] = bool(verdict.get("pass"))
    state["vision_score"] = float(verdict.get("score", 0.0))
    state["vision_critique"] = verdict.get("critique", "")

    sig = state.get("signature")
    if sig and not state["vision_pass"]:
        TemplateMetrics().record_vision_fail(sig)

    return state



def should_pass_or_review(state) -> str:
    if state.get("vision_pass", False) and float(state.get("vision_score", 0.0)) >= 0.7:
        return "pass"
    if int(state.get("refine_attempts", 0)) >= 1:
        return "review_final"
    return "review"



import os
from src.invoice.template_cache import TemplateCache
from src.invoice.metrics import TemplateMetrics
from src.invoice.doc_vlm_extract import extract_with_doc_vlm

DOCVLM_DEBUG = os.getenv('DOCVLM_DEBUG', 'false').lower() == 'true'
from src.invoice.cerbos_client import can_promote_template

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
from src.invoice.doc_vlm_extract import extract_with_doc_vlm

DOCVLM_DEBUG = os.getenv('DOCVLM_DEBUG', 'false').lower() == 'true'
from src.invoice.cerbos_client import can_promote_template

AUTO_PROMOTE_THRESHOLD = int(os.getenv("AUTO_PROMOTE_THRESHOLD", "3"))


def node_promote_template(state):
    sig = state.get("signature")
    stage = state.get("template_source", "staging")
    role = state.get("role") or os.getenv("APP_ROLE", "employee")

    cache = TemplateCache()
    metrics = TemplateMetrics()

    # Check metrics
    m = metrics.get(sig) if sig else {}
    success_count = m.get("success_count", 0)

    if success_count < AUTO_PROMOTE_THRESHOLD:
        # Not enough successful runs; force manual review path
        state["promotion_status"] = f"pending_success_{success_count}"
        return state

    # Now enough successes; ask Cerbos
    allowed = can_promote_template(role=role, stage=stage)

    if not allowed:
        state["promotion_status"] = "denied"
        return state

    # Promote in cache
    promoted = cache.promote(sig)
    if promoted:
        state["template_source"] = "active"
        state["promotion_status"] = "promoted"
        metrics.record_promotion(sig)
    else:
        state["promotion_status"] = "promote_failed"

    return state




def node_done(state):
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
