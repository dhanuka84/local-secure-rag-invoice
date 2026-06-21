import os
import time

from langgraph.graph import StateGraph, START, END

from .state import InvoiceState
from .nodes import (
    # Core extraction
    node_extract_pdf,
    node_ocr_if_needed,
    node_structure_detect,
    node_signature,
    node_check_cache,
    # Decisions
    should_reuse_or_search,
    node_milvus_suggest,
    should_use_suggest_or_learn,
    # Template-based extraction
    node_learn_and_stage,   # now actually used
    node_extract_fields,
    # LayoutLM hybrid extraction (no Donut)
    node_hybrid_extract_fields,
    # Vision + promotion
    node_vision_validate,
    should_pass_or_review,
    node_promote_template,
    node_mark_for_review,
    node_done,
)

TRACE_NODES = os.getenv("INVOICE_TRACE_NODES", "true").lower() == "true"


def _timed_node(name, fn):
    def wrapped(state):
        started = time.perf_counter()
        if TRACE_NODES:
            print(f"[graph] start {name}")
        out = fn(state)
        elapsed = time.perf_counter() - started
        if isinstance(out, dict):
            timings = dict(out.get("node_timings") or {})
            timings[name] = round(elapsed, 3)
            out["node_timings"] = timings
        if TRACE_NODES:
            print(f"[graph] done {name} in {elapsed:.3f}s")
        return out

    return wrapped


def build_invoice_graph():
    """
    Build and compile the invoice processing graph.

    High-level flow:

        START
          -> extract_pdf
          -> ocr_if_needed
          -> signature
          -> check_cache
          -> (reuse | search via milvus_suggest)

        If reuse:
            -> extract_fields
            -> hybrid_extract_fields
            -> vision_validate

        If search:
            -> milvus_suggest
                -> (suggested | learn)

            If suggested:
                -> extract_fields
                -> hybrid_extract_fields
                -> vision_validate

            If learn:
                -> learn_and_stage       (learn regex template)
                -> extract_fields
                -> hybrid_extract_fields
                -> vision_validate

        vision_validate
          -> (promote_template | mark_for_review)
          -> done
          -> END
    """

    g = StateGraph(InvoiceState)

    # ---------- Nodes ----------

    # Core pipeline
    g.add_node("extract_pdf", _timed_node("extract_pdf", node_extract_pdf))
    g.add_node("ocr_if_needed", _timed_node("ocr_if_needed", node_ocr_if_needed))
    g.add_node("structure_detect", _timed_node("structure_detect", node_structure_detect))
    g.add_node("signature", _timed_node("signature", node_signature))
    g.add_node("check_cache", _timed_node("check_cache", node_check_cache))

    # Similarity / search
    g.add_node("milvus_suggest", _timed_node("milvus_suggest", node_milvus_suggest))

    # Template-based extraction
    g.add_node("extract_fields", _timed_node("extract_fields", node_extract_fields))
    g.add_node("learn_and_stage", _timed_node("learn_and_stage", node_learn_and_stage))

    # LayoutLM hybrid extraction (no Donut here)
    g.add_node("hybrid_extract_fields", _timed_node("hybrid_extract_fields", node_hybrid_extract_fields))

    # Vision model
    g.add_node("vision_validate", _timed_node("vision_validate", node_vision_validate))

    # Promotion / review
    g.add_node("promote_template", _timed_node("promote_template", node_promote_template))
    g.add_node("mark_for_review", _timed_node("mark_for_review", node_mark_for_review))

    # Final
    g.add_node("done", _timed_node("done", node_done))

    # ---------- Edges ----------

    # Entry point
    g.add_edge(START, "extract_pdf")
    g.add_edge("extract_pdf", "ocr_if_needed")
    g.add_edge("ocr_if_needed", "structure_detect")
    g.add_edge("structure_detect", "signature")
    g.add_edge("signature", "check_cache")

    # Reuse vs search
    g.add_conditional_edges(
        "check_cache",
        should_reuse_or_search,  # returns "reuse" or "search"
        {
            "reuse": "extract_fields",
            "search": "milvus_suggest",
        },
    )

    # Suggested template vs learn new template
    g.add_conditional_edges(
        "milvus_suggest",
        should_use_suggest_or_learn,  # returns "suggested" or "learn"
        {
            "suggested": "extract_fields",
            "learn": "learn_and_stage",
        },
    )

    # Learn path: learn template then extract
    g.add_edge("learn_and_stage", "extract_fields")

    # After regex extraction, go directly to LayoutLM hybrid (NO doc_vlm)
    g.add_edge("extract_fields", "hybrid_extract_fields")

    # Hybrid → vision
    g.add_edge("hybrid_extract_fields", "vision_validate")

    # Vision → promote vs review
    g.add_conditional_edges(
        "vision_validate",
        should_pass_or_review,  # returns "pass" or "review"
        {
            "pass": "promote_template",
            "review": "mark_for_review",
        },
    )

    # Both paths converge on done
    g.add_edge("promote_template", "done")
    g.add_edge("mark_for_review", "done")

    # End the graph
    g.add_edge("done", END)

    return g.compile()
