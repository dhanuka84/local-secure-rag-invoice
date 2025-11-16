
import json
import time
from typing import Optional, Dict, List
import redis

ACTIVE_KEY   = "invoice:template:{signature}"
STAGING_KEY  = "invoice:template:staging:{signature}"

class TemplateCache:
    def __init__(self, host="127.0.0.1", port=6379, staging_ttl_seconds: int = 7 * 24 * 3600):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.staging_ttl = staging_ttl_seconds

    def _k_active(self, signature: str) -> str:
        return ACTIVE_KEY.format(signature=signature)

    def _k_staging(self, signature: str) -> str:
        return STAGING_KEY.format(signature=signature)

    def get_active(self, signature: str) -> Optional[Dict]:
        v = self.r.get(self._k_active(signature))
        return json.loads(v) if v else None

    def get_staging(self, signature: str) -> Optional[Dict]:
        v = self.r.get(self._k_staging(signature))
        return json.loads(v) if v else None

    def set_active(self, signature: str, template: Dict):
        t = dict(template)
        t.setdefault("version", int(time.time()))
        t.setdefault("status", "active")
        self.r.set(self._k_active(signature), json.dumps(t))

    def set_staging(self, signature: str, template: Dict):
        t = dict(template)
        t.setdefault("version", int(time.time()))
        t["status"] = "staging"
        self.r.set(self._k_staging(signature), json.dumps(t), ex=self.staging_ttl)

    def list_active(self) -> List[str]:
        return [k.split("invoice:template:")[-1]
                for k in self.r.keys("invoice:template:*")
                if not k.startswith("invoice:template:staging:")]

    def list_staging(self) -> List[str]:
        return [k.split("invoice:template:staging:")[-1]
                for k in self.r.keys("invoice:template:staging:*")]

    def promote(self, signature: str) -> bool:
        s_key = self._k_staging(signature)
        a_key = self._k_active(signature)
        v = self.r.get(s_key)
        if not v:
            return False
        t = json.loads(v)
        t["status"] = "active"
        t.setdefault("promoted_at", int(time.time()))
        self.r.set(a_key, json.dumps(t))
        self.r.delete(s_key)
        return True

    def reject(self, signature: str) -> bool:
        return bool(self.r.delete(self._k_staging(signature)))

    def remove_active(self, signature: str) -> bool:
        return bool(self.r.delete(self._k_active(signature)))
