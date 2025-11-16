
from src.invoice.pdf_io import pdf_to_text_and_images, ocr_if_needed
from src.invoice.signature import build_signature
from src.invoice.template_cache import TemplateCache
from src.invoice.extract import extract_fields
from src.invoice.template_learner import learn_regexes
from src.invoice.vision_validate import validate_with_vision
from src.invoice.cerbos_client import can_promote_template
from src.invoice.metrics import TemplateMetrics
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



def node_promote_template(state):
    """Promote template from learned/staging/refined to active when vision passed and Cerbos allows."""
    from src.invoice.template_cache import TemplateCache

    role = state.get("role", "employee")
    source = state.get("template_source", "")
    signature = state.get("signature", "")
    vision_pass = bool(state.get("vision_pass", False))
    vision_score = float(state.get("vision_score", 0.0))

    if not (vision_pass and vision_score >= 0.7):
        state["promotion_status"] = "skipped_vision"
        return state

    if source not in ("learned", "staging", "refined"):
        state["promotion_status"] = "skipped_source"
        return state

    allowed = can_promote_template(role, stage="staging")
    if not allowed:
        state["promotion_status"] = "denied"
        return state

    cache = TemplateCache()
    if cache.promote(signature):
        state["template_source"] = "active"
        state["promotion_status"] = "promoted"
        TemplateMetrics().record_promotion(signature)
    else:
        state["promotion_status"] = "no_staging_template"

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
