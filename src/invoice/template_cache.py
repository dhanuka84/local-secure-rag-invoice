import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import redis

ACTIVE_KEY = "invoice:template:{signature}"
STAGING_KEY = "invoice:template:staging:{signature}"
LOCAL_STATE_DIR = Path(os.getenv("INVOICE_LOCAL_STATE_DIR", Path(tempfile.gettempdir()) / "local-secure-rag-invoice"))
LOCAL_CACHE_FILE = LOCAL_STATE_DIR / "template_cache.json"
REDIS_CONNECT_TIMEOUT = float(os.getenv("INVOICE_REDIS_CONNECT_TIMEOUT_SECONDS", "0.25"))


class TemplateCache:
    def __init__(self, host="127.0.0.1", port=6379, staging_ttl_seconds: int = 7 * 24 * 3600):
        self.host = host
        self.port = port
        self.staging_ttl = staging_ttl_seconds
        self.r = redis.Redis(
            host=host,
            port=port,
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
            print(f"[template_cache] Redis unavailable at {self.host}:{self.port}; using local file cache. {exc}")
            return False

    def _load_local(self) -> Dict[str, Dict[str, Dict]]:
        if not LOCAL_CACHE_FILE.exists():
            return {"active": {}, "staging": {}}
        try:
            return json.loads(LOCAL_CACHE_FILE.read_text())
        except Exception:
            return {"active": {}, "staging": {}}

    def _save_local(self, data: Dict[str, Dict[str, Dict]]):
        LOCAL_CACHE_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))

    def _prune_expired_local(self, data: Dict[str, Dict[str, Dict]]):
        now = int(time.time())
        staging = data.setdefault("staging", {})
        expired = [sig for sig, value in staging.items() if value.get("_expires_at", now + 1) <= now]
        for sig in expired:
            staging.pop(sig, None)

    def _k_active(self, signature: str) -> str:
        return ACTIVE_KEY.format(signature=signature)

    def _k_staging(self, signature: str) -> str:
        return STAGING_KEY.format(signature=signature)

    def get_active(self, signature: str) -> Optional[Dict]:
        if self._redis_available:
            v = self.r.get(self._k_active(signature))
            return json.loads(v) if v else None
        data = self._load_local()
        self._prune_expired_local(data)
        return data.get("active", {}).get(signature)

    def get_staging(self, signature: str) -> Optional[Dict]:
        if self._redis_available:
            v = self.r.get(self._k_staging(signature))
            return json.loads(v) if v else None
        data = self._load_local()
        self._prune_expired_local(data)
        return data.get("staging", {}).get(signature)

    def set_active(self, signature: str, template: Dict):
        t = dict(template)
        t.setdefault("version", int(time.time()))
        t.setdefault("status", "active")
        if self._redis_available:
            self.r.set(self._k_active(signature), json.dumps(t))
            return
        data = self._load_local()
        data.setdefault("active", {})[signature] = t
        self._save_local(data)

    def set_staging(self, signature: str, template: Dict):
        t = dict(template)
        t.setdefault("version", int(time.time()))
        t["status"] = "staging"
        if self._redis_available:
            self.r.set(self._k_staging(signature), json.dumps(t), ex=self.staging_ttl)
            return
        t["_expires_at"] = int(time.time()) + self.staging_ttl
        data = self._load_local()
        data.setdefault("staging", {})[signature] = t
        self._save_local(data)

    def list_active(self) -> List[str]:
        if self._redis_available:
            return [
                key.split("invoice:template:")[-1]
                for key in self.r.keys("invoice:template:*")
                if not key.startswith("invoice:template:staging:")
            ]
        data = self._load_local()
        return list(data.get("active", {}).keys())

    def list_staging(self) -> List[str]:
        if self._redis_available:
            return [key.split("invoice:template:staging:")[-1] for key in self.r.keys("invoice:template:staging:*")]
        data = self._load_local()
        self._prune_expired_local(data)
        self._save_local(data)
        return list(data.get("staging", {}).keys())

    def promote(self, signature: str) -> bool:
        if self._redis_available:
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

        data = self._load_local()
        self._prune_expired_local(data)
        template = data.setdefault("staging", {}).pop(signature, None)
        if not template:
            return False
        template.pop("_expires_at", None)
        template["status"] = "active"
        template.setdefault("promoted_at", int(time.time()))
        data.setdefault("active", {})[signature] = template
        self._save_local(data)
        return True

    def reject(self, signature: str) -> bool:
        if self._redis_available:
            return bool(self.r.delete(self._k_staging(signature)))
        data = self._load_local()
        removed = data.setdefault("staging", {}).pop(signature, None) is not None
        self._save_local(data)
        return removed

    def remove_active(self, signature: str) -> bool:
        if self._redis_available:
            return bool(self.r.delete(self._k_active(signature)))
        data = self._load_local()
        removed = data.setdefault("active", {}).pop(signature, None) is not None
        self._save_local(data)
        return removed
