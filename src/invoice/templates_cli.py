
import json
import argparse
from src.invoice.template_cache import TemplateCache
from src.invoice.cerbos_client import can_promote_template

def cmd_list(args):
    c = TemplateCache()
    print("Active:")
    for s in sorted(c.list_active()):
        print("  ", s)
    print("\nStaging:")
    for s in sorted(c.list_staging()):
        print("  ", s)

def cmd_show(args):
    c = TemplateCache()
    t = c.get_active(args.signature) or c.get_staging(args.signature)
    if not t:
        print("Not found.")
        return
    print(json.dumps(t, indent=2))

def cmd_promote(args):
    c = TemplateCache()
    if not can_promote_template(args.role, stage="staging"):
        print("Denied by Cerbos.")
        return
    ok = c.promote(args.signature)
    print("Promoted." if ok else "No staging template to promote.")

def cmd_reject(args):
    c = TemplateCache()
    ok = c.reject(args.signature)
    print("Rejected (deleted from staging)." if ok else "Not found in staging.")

def cmd_remove_active(args):
    c = TemplateCache()
    ok = c.remove_active(args.signature)
    print("Removed active." if ok else "Active not found.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="invoice-templates")
    sub = ap.add_subparsers(dest="cmd")

    sp = sub.add_parser("list", help="List active & staging")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("show", help="Show a template (active or staging)")
    sp.add_argument("signature")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("promote", help="Promote staging -> active (requires Cerbos allow)")
    sp.add_argument("signature")
    sp.add_argument("--role", default="manager")
    sp.set_defaults(func=cmd_promote)

    sp = sub.add_parser("reject", help="Reject (delete) from staging")
    sp.add_argument("signature")
    sp.set_defaults(func=cmd_reject)

    sp = sub.add_parser("remove-active", help="Delete from active")
    sp.add_argument("signature")
    sp.set_defaults(func=cmd_remove_active)

    args = ap.parse_args()
    if not getattr(args, "cmd", None):
        ap.print_help()
    else:
        args.func(args)
