import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOCAL_STATE_DIR = Path(os.getenv("INVOICE_LOCAL_STATE_DIR", Path(tempfile.gettempdir()) / "local-secure-rag-invoice"))
LOCAL_METRICS_FILE = LOCAL_STATE_DIR / "metrics.json"
REDIS_CONNECT_TIMEOUT = float(os.getenv("INVOICE_REDIS_CONNECT_TIMEOUT_SECONDS", "0.25"))


class TemplateMetrics:
    """Redis-backed metrics with a local file fallback."""

    def __init__(self):
        self.r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
            socket_timeout=REDIS_CONNECT_TIMEOUT,
        )
        self._redis_available = self._can_use_redis()
        LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _can_use_redis(self) -> bool:
        try:
            self.r.ping()
            return True
        except Exception as exc:
            print(f"[metrics] Redis unavailable at {REDIS_HOST}:{REDIS_PORT}; using local file metrics. {exc}")
            return False

    def _load_local(self) -> Dict[str, Dict]:
        if not LOCAL_METRICS_FILE.exists():
            return {}
        try:
            return json.loads(LOCAL_METRICS_FILE.read_text())
        except Exception:
            return {}

    def _save_local(self, data: Dict[str, Dict]):
        LOCAL_METRICS_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))

    def _key(self, signature: str) -> str:
        return f"invoice_metrics:{signature}"

    def _bump(self, signature: str, field: str):
        if self._redis_available:
            k = self._key(signature)
            pipe = self.r.pipeline()
            pipe.hincrby(k, field, 1)
            pipe.hset(k, "updated_at", int(time.time()))
            pipe.execute()
            return

        data = self._load_local()
        entry = data.setdefault(signature, {})
        entry[field] = int(entry.get(field, 0)) + 1
        entry["updated_at"] = int(time.time())
        self._save_local(data)

    def record_refine(self, signature: str):
        self._bump(signature, "refine_attempts")

    def record_promotion(self, signature: str):
        self._bump(signature, "promotions")

    def record_vision_fail(self, signature: str):
        self._bump(signature, "vision_failures")

    def record_success(self, signature: str):
        self._bump(signature, "success_count")

    def get(self, signature: str) -> Dict:
        if self._redis_available:
            k = self._key(signature)
            data = self.r.hgetall(k)
            if not data:
                return {}
            out: Dict = {}
            for key, value in data.items():
                if key in {"refine_attempts", "promotions", "vision_failures", "success_count"}:
                    out[key] = int(value)
                else:
                    out[key] = value
            return out
        return self._load_local().get(signature, {})

    def list_all(self) -> Dict[str, Dict]:
        if self._redis_available:
            out: Dict[str, Dict] = {}
            for key in self.r.keys("invoice_metrics:*"):
                sig = key.split("invoice_metrics:")[-1]
                out[sig] = self.get(sig)
            return out
        return self._load_local()


if __name__ == "__main__":
    tm = TemplateMetrics()
    print(json.dumps(tm.list_all(), indent=2))
