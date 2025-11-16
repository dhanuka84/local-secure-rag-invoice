# src/tools/inspect_templates.py
import os
import json
import redis
from datetime import datetime

from src.invoice.template_cache import TemplateCache
from src.invoice.metrics import TemplateMetrics

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


def _ts(ts_str: str | None) -> str:
    if not ts_str:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts_str)).isoformat(timespec="seconds")
    except Exception:
        return ts_str


def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    cache = TemplateCache()
    metrics = TemplateMetrics()

    # Discover all template keys
    keys = r.keys("invoice:template:*")
    if not keys:
        print("No templates found in Redis.")
        return

    print("\n=== Invoice Template Evolution ===\n")

    for k in sorted(keys):
        sig = k.split("invoice:template:", 1)[1]
        tmpl = cache.get(sig)  # however your TemplateCache exposes it
        m = metrics.get(sig)

        print(f"Signature : {sig}")
        print(f"Stage     : {tmpl.get('stage', 'unknown') if tmpl else 'unknown'}")
        print(f"Created   : {_ts(tmpl.get('created_at')) if tmpl else '-'}")
        print(f"Updated   : {_ts(tmpl.get('updated_at')) if tmpl else '-'}")

        print("Metrics   :")
        print(f"  refine_attempts : {m.get('refine_attempts', 0)}")
        print(f"  promotions      : {m.get('promotions', 0)}")
        print(f"  vision_failures : {m.get('vision_failures', 0)}")
        print(f"  last_metrics_ts : {_ts(m.get('updated_at'))}")

        # Optional: show fields the template cares about
        fields = tmpl.get("fields", {}) if tmpl else {}
        if fields:
            print("Fields    :")
            for name, pattern in fields.items():
                preview = str(pattern)[:80].replace("\n", " ")
                print(f"  - {name}: {preview}")
        print("-" * 60)


if __name__ == "__main__":
    main()
