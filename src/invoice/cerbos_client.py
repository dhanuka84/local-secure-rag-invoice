
import os
from cerbos.sdk.client import CerbosClient
from cerbos.sdk.model import Principal, Resource

CERBOS_URL = os.getenv("CERBOS_URL", "http://localhost:3592")

def can_promote_template(role: str, stage: str = "staging") -> bool:
    client = CerbosClient(host=CERBOS_URL)
    principal = Principal(id="user", roles=[role])
    res = Resource(id="template", kind="template", attr={"stage": stage})
    try:
        decision = client.check_resource(principal=principal, resource=res, actions={"promote"})
        if hasattr(decision, "is_allowed"):
            return decision.is_allowed("promote")
        actions = getattr(decision, "actions", {})
        return bool(actions.get("promote", False))
    except Exception:
        return False
