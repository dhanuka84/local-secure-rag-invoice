import json
import argparse
import os
import sys
import faulthandler
import signal

from .build import build_invoice_graph


# --- Timeout + stack dump for debugging hangs ---
def dump_stack(signum, frame):
    print("\n\n=== TIMEOUT: dumping stack ===", file=sys.stderr)
    faulthandler.dump_traceback(file=sys.stderr)


faulthandler.enable()
signal.signal(signal.SIGALRM, dump_stack)
# 120-second hard timeout for the whole run
signal.alarm(120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to invoice PDF")
    args = ap.parse_args()

    role = os.getenv("APP_ROLE", "employee")

    graph = build_invoice_graph()

    # IMPORTANT: pass pdf_path into the initial state
    state = {
        "pdf_path": args.pdf,
        "role": role,
    }

    out = graph.invoke(state)

    print(
        json.dumps(
            {
                "pdf": args.pdf,
                "signature": out.get("signature"),
                "template_source": out.get("template_source"),
                "promotion_status": out.get("promotion_status"),
                "fields": out.get("fields"),
                "vision_pass": out.get("vision_pass"),
                "vision_score": out.get("vision_score"),
                "vision_critique": out.get("vision_critique"),
                "done": out.get("done"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
