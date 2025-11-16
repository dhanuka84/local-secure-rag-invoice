from langgraph.graph import StateGraph, END
from .state import InvoiceState
from .nodes import (
    node_extract_pdf,
    node_ocr_if_needed,
    node_signature,
    node_check_cache,
    should_reuse_or_search,
    node_milvus_suggest,
    should_use_suggest_or_learn,
    node_learn_and_stage,
    node_extract_fields,
    node_vision_validate,
    should_pass_or_review,
    node_promote_template,
    node_done,
    node_mark_for_review,
)


def build_invoice_graph():
    """
    Build and compile the invoice processing graph.

    High-level flow:

        extract_pdf
          -> ocr_if_needed
          -> signature
          -> check_cache ---[reuse/search]--> extract_fields OR milvus_suggest
          -> (maybe) learn_and_stage
          -> extract_fields
          -> vision_validate ---[pass/review]--> promote_template / mark_for_review
          -> done
    """

    g = StateGraph(InvoiceState)

    # ---------- Nodes ----------
    g.add_node("extract_pdf",       node_extract_pdf)
    g.add_node("ocr_if_needed",     node_ocr_if_needed)
    g.add_node("signature",         node_signature)
    g.add_node("check_cache",       node_check_cache)

    g.add_node("milvus_suggest",    node_milvus_suggest)
    g.add_node("learn_and_stage",   node_learn_and_stage)
    g.add_node("extract_fields",    node_extract_fields)
    g.add_node("vision_validate",   node_vision_validate)

    g.add_node("promote_template",  node_promote_template)
    g.add_node("mark_for_review",   node_mark_for_review)
    g.add_node("done",              node_done)

    # ---------- Linear skeleton ----------
    g.set_entry_point("extract_pdf")
    g.add_edge("extract_pdf", "ocr_if_needed")
    g.add_edge("ocr_if_needed", "signature")
    g.add_edge("signature", "check_cache")

    # ---------- Cache decision ----------
    # should_reuse_or_search(state) → "reuse" or "search"
    g.add_conditional_edges(
        "check_cache",
        should_reuse_or_search,
        {
            # We already have a good template / fields: just extract & go on
            "reuse": "extract_fields",
            # Need to search / learn a template via Milvus
            "search": "milvus_suggest",
        },
    )

    # ---------- Milvus suggestion decision ----------
    # should_use_suggest_or_learn(state) → "use_suggest" or "learn"
    g.add_conditional_edges(
        "milvus_suggest",
        should_use_suggest_or_learn,
        {
            # Use suggested template directly
            "use_suggest": "extract_fields",
            # Need to learn a new template and stage it
            "learn": "learn_and_stage",
        },
    )

    # After learning a new template, apply it via extract_fields
    g.add_edge("learn_and_stage", "extract_fields")

    # ---------- Extraction → vision ----------
    g.add_edge("extract_fields", "vision_validate")

    # ---------- Vision / policy decision ----------
    # should_pass_or_review(state) → "pass" or "review"
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

    # End of graph
    g.add_edge("done", END)

    # IMPORTANT: return compiled graph (has .invoke)
    return g.compile()
