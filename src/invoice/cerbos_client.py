import os
from cerbos.sdk.client import CerbosClient
from cerbos.sdk.model import Principal, Resource

CERBOS_HOST = os.getenv("CERBOS_HOST", "http://localhost:3592")
CERBOS_STRICT = os.getenv("CERBOS_STRICT", "false").lower() == "true"

_client = CerbosClient(host=CERBOS_HOST)


def can_promote_template(role: str, stage: str) -> bool:
    """
    Ask Cerbos whether this role can promote a template in this stage.
    Uses is_allowed(), which is available in all recent Python SDK versions.
    """

    principal = Principal(
        id="user",
        roles=[role],
    )

    resource = Resource(
        id="invoice_template",
        kind="template",
        attr={"stage": stage},
    )

    try:
        # Recommended modern API:
        allowed = _client.is_allowed("promote", principal, resource)
        return bool(allowed)

    except Exception as e:
        print(f"[CERBOS] Error while checking promote: {e}")

        # Strict mode: deny if Cerbos is unreachable or errors
        if CERBOS_STRICT:
            return False

        # Non-strict mode: allow on failure (developer-friendly)
        return True

