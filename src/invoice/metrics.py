
import os
import json
import time
from typing import Dict
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


class TemplateMetrics:
    """Small Redis-backed metrics helper for invoice templates."""

    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    def _key(self, signature: str) -> str:
        return f"invoice_metrics:{signature}"

    def _bump(self, signature: str, field: str):
        k = self._key(signature)
        pipe = self.r.pipeline()
        pipe.hincrby(k, field, 1)
        pipe.hset(k, "updated_at", int(time.time()))
        pipe.execute()

    def record_refine(self, signature: str):
        self._bump(signature, "refine_attempts")

    def record_promotion(self, signature: str):
        self._bump(signature, "promotions")

    def record_vision_fail(self, signature: str):
        self._bump(signature, "vision_failures")

    def get(self, signature: str) -> Dict:
        k = self._key(signature)
        data = self.r.hgetall(k)
        if not data:
            return {}
        out: Dict = {}
        for k2, v in data.items():
            if k2 in {"refine_attempts", "promotions", "vision_failures"}:
                out[k2] = int(v)
            else:
                out[k2] = v
        return out

    def list_all(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for key in self.r.keys("invoice_metrics:*"):
            sig = key.split("invoice_metrics:")[-1]
            out[sig] = self.get(sig)
        return out
    
    
    def record_success(self, signature: str):
        self._bump(signature, "success_count")



if __name__ == "__main__":
    tm = TemplateMetrics()
    print(json.dumps(tm.list_all(), indent=2))
