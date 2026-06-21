
import json, argparse, os
from .build import build_invoice_graph
from src.invoice.ollama_runtime import summarize_ollama_runtime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to invoice PDF")
    args = ap.parse_args()

    role = os.getenv("APP_ROLE", "employee")

    graph = build_invoice_graph()
    state = {"pdf_path": args.pdf, "role": role, "ollama_runtime": summarize_ollama_runtime()}
    out = graph.invoke(state)

    print(json.dumps({
        "pdf": args.pdf,
        "signature": out.get("signature"),
        "template_source": out.get("template_source"),
        "template_resolution_mode": out.get("template_resolution_mode"),
        "field_extraction_backend": out.get("field_extraction_backend"),
        "template_learning_backend": out.get("template_learning_backend"),
        "template_learning_model": out.get("template_learning_model"),
        "embedding_backend": out.get("embedding_backend"),
        "embedding_model": out.get("embedding_model"),
        "promotion_status": out.get("promotion_status"),
        "fields": out.get("fields"),
        "validation_mode": out.get("validation_mode"),
        "audit_warning": out.get("audit_warning"),
        "audit_reason": out.get("audit_reason"),
        "vision_backend": out.get("vision_backend"),
        "vision_model": out.get("vision_model"),
        "vision_pass": out.get("vision_pass"),
        "vision_score": out.get("vision_score"),
        "vision_critique": out.get("vision_critique"),
        "ollama_runtime": out.get("ollama_runtime"),
        "done": out.get("done"),
    }, indent=2, default=str))

if __name__ == "__main__":
    main()
