# src/tools/batch_run_invoices.py
import os
import glob
import json
from typing import List

from src.graph_invoice.build import build_invoice_graph
from src.graph_invoice.state import InvoiceState


def run_one(graph, pdf_path: str, role: str = "manager") -> dict:
    state = InvoiceState(
        pdf_path=pdf_path,
        role=role,
    )
    out = graph.invoke(state)
    # out is usually a dict-like or InvoiceState; adapt if needed
    return dict(out)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="samples/invoices",
        help="Directory containing invoice PDFs",
    )
    parser.add_argument(
        "--role",
        default=os.getenv("APP_ROLE", "manager"),
        help="Role to use (manager/employee)",
    )
    args = parser.parse_args()

    pdfs: List[str] = sorted(glob.glob(os.path.join(args.dir, "*.pdf")))
    if not pdfs:
        print(f"No PDFs found in {args.dir}")
        return

    graph = build_invoice_graph()

    summary = []
    print(f"\n=== Batch run: {len(pdfs)} invoices, role={args.role} ===\n")
    for pdf in pdfs:
        print(f"--> {pdf}")
        result = run_one(graph, pdf, role=args.role)

        sig = result.get("signature")
        stage = result.get("template_source")
        promoted = result.get("promotion_status")
        vpass = result.get("vision_pass")
        vscore = result.get("vision_score")

        print(f"    signature        : {sig}")
        print(f"    template_source  : {stage}")
        print(f"    promotion_status : {promoted}")
        print(f"    vision_pass/score: {vpass} / {vscore}")
        print()

        summary.append(
            {
                "pdf": pdf,
                "signature": sig,
                "stage": stage,
                "promotion_status": promoted,
                "vision_pass": vpass,
                "vision_score": vscore,
            }
        )

    print("\n=== Summary (JSON) ===\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
