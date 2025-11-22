from langgraph.graph import StateGraph, END

from .state import InvoiceState
from .nodes import (
    # Core extraction
    node_extract_pdf,
    node_ocr_if_needed,
    node_signature,
    node_check_cache,
    # Decisions
    should_reuse_or_search,
    node_milvus_suggest,
    should_use_suggest_or_learn,
    # Template-based extraction
    node_learn_and_stage,   # currently unused in this graph, safe to keep
    node_extract_fields,
    # Donut + LayoutLM hybrid extraction
    node_doc_vlm_extract_fields,
    node_hybrid_extract_fields,
    # Vision + promotion
    node_vision_validate,
    should_pass_or_review,
    node_promote_template,
    node_mark_for_review,
    node_done,
)


def build_invoice_graph():
    """
    Build and compile the invoice processing graph.

    High-level flow:

        extract_pdf
          -> ocr_if_needed
          -> signature
          -> check_cache
          -> (reuse | search via milvus_suggest)

        If reuse:
            -> extract_fields
            -> vision_validate
        If search:
            -> milvus_suggest
                -> (suggested | learn)

            If suggested:
                -> extract_fields
                -> vision_validate

            If learn:
                -> doc_vlm_extract_fields  (Donut)
                -> hybrid_extract_fields   (Donut + LayoutLM)
                -> vision_validate

        vision_validate
          -> (promote_template | mark_for_review)
          -> done
    """

    g = StateGraph(InvoiceState)

    # ---------- Nodes ----------

    # Core pipeline
    g.add_node("extract_pdf", node_extract_pdf)
    g.add_node("ocr_if_needed", node_ocr_if_needed)
    g.add_node("signature", node_signature)
    g.add_node("check_cache", node_check_cache)

    # Similarity / search
    g.add_node("milvus_suggest", node_milvus_suggest)

    # Template-based extraction
    g.add_node("extract_fields", node_extract_fields)

    # Donut + LayoutLM hybrid extraction
    g.add_node("doc_vlm_extract_fields", node_doc_vlm_extract_fields)
    g.add_node("hybrid_extract_fields", node_hybrid_extract_fields)

    # Vision model
    g.add_node("vision_validate", node_vision_validate)

    # Promotion / review
    g.add_node("promote_template", node_promote_template)
    g.add_node("mark_for_review", node_mark_for_review)

    # Final
    g.add_node("done", node_done)

    # ---------- Entry & linear spine ----------

    g.set_entry_point("extract_pdf")
    g.add_edge("extract_pdf", "ocr_if_needed")
    g.add_edge("ocr_if_needed", "signature")
    g.add_edge("signature", "check_cache")

    # ---------- Cache decision: reuse vs search ----------

    # should_reuse_or_search(state) -> "reuse" | "search"
    g.add_conditional_edges(
        "check_cache",
        should_reuse_or_search,
        {
            # Reuse existing ACTIVE template (fast path)
            "reuse": "extract_fields",
            # No active template: go via milvus + Donut/LayoutLM
            "search": "milvus_suggest",
        },
    )

    # ---------- Milvus decision: suggested vs learn ----------

    # should_use_suggest_or_learn(state) -> "suggested" | "learn"
    g.add_conditional_edges(
        "milvus_suggest",
        should_use_suggest_or_learn,
        {
            # Use suggested template (template-based path)
            "suggested": "extract_fields",
            # Unknown layout → Donut + LayoutLM hybrid extraction
            "learn": "doc_vlm_extract_fields",
        },
    )

    # Donut → hybrid (ensemble of Donut + LayoutLM)
    g.add_edge("doc_vlm_extract_fields", "hybrid_extract_fields")

    # Template path → vision
    g.add_edge("extract_fields", "vision_validate")

    # Hybrid path → vision
    g.add_edge("hybrid_extract_fields", "vision_validate")

    # ---------- Vision / policy decision ----------

    # should_pass_or_review(state) -> "pass" | "review"
    g.add_conditional_edges(
        "vision_validate",
        should_pass_or_review,
        {
            "pass": "promote_template",
            "review": "mark_for_review",
        },
    )

    # ---------- Finalization ----------

    g.add_edge("promote_template", "done")
    g.add_edge("mark_for_review", "done")
    g.add_edge("done", END)

    return g.compile()
