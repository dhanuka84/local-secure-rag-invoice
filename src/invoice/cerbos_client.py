import os
from cerbos.sdk.client import CerbosClient
from cerbos.sdk.model import Principal, Resource

CERBOS_HOST = os.getenv("CERBOS_HOST", "http://localhost:3592")
CERBOS_STRICT = os.getenv("CERBOS_STRICT", "false").lower() == "true"

_client = CerbosClient(host=CERBOS_HOST)


def can_promote_template(role: str, stage: str) -> bool:
    """
    Ask Cerbos: can this role promote a template in this stage?

    Matches policy:
      resource: "template"
      action:  "promote"
      attr:    { "stage": <stage> }
    """
    principal = Principal(
        id="user",
        roles=[role],           # must be ["manager"] or ["employee"]
    )

    resource = Resource(
        id="invoice_template",  # arbitrary ID, not used in the policy
        kind="template",        # MUST match resource: "template" in YAML
        attr={"stage": stage},  # MUST match condition on request.resource.attr.stage
    )

    try:
        decision = _client.check_resource(
            principal=principal,
            resource=resource,
            actions={"promote"},
        )

        # Newer SDK: decision.is_allowed("promote")
        if hasattr(decision, "is_allowed") and callable(decision.is_allowed):
            return decision.is_allowed("promote")

        # Fallback: look at decision.actions dict
        actions = getattr(decision, "actions", None)
        if isinstance(actions, dict):
            return bool(actions.get("promote"))

        # If we couldn't interpret the decision, be strict or permissive
        return False if CERBOS_STRICT else True

    except Exception as e:
        print(f"[CERBOS] Error while checking promote: {e}")
        # In dev, you can choose to allow on failure
        return False if CERBOS_STRICT else True
