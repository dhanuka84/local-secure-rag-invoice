
import json, argparse, os
from .build import build_invoice_graph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to invoice PDF")
    args = ap.parse_args()

    role = os.getenv("APP_ROLE", "employee")

    graph = build_invoice_graph()
    state = {"pdf_path": args.pdf, "role": role}
    out = graph.invoke(state)

    print(json.dumps({
        "pdf": args.pdf,
        "signature": out.get("signature"),
        "template_source": out.get("template_source"),
        "promotion_status": out.get("promotion_status"),
        "fields": out.get("fields"),
        "vision_pass": out.get("vision_pass"),
        "vision_score": out.get("vision_score"),
        "vision_critique": out.get("vision_critique"),
        "done": out.get("done"),
    }, indent=2, default=str))

if __name__ == "__main__":
    main()
