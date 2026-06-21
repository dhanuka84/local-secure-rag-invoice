import os
from cerbos.sdk.client import CerbosClient
from cerbos.sdk.model import (
    Principal,
    Resource,
    ResourceAction,
    ResourceList,
)

CERBOS_HOST = os.getenv("CERBOS_HOST", "localhost:3593")
CERBOS_STRICT = os.getenv("CERBOS_STRICT", "false").lower() == "true"



def can_promote_template(role: str, stage: str) -> bool:
    """
    Ask Cerbos if this role can promote an invoice template in the given stage.
    Uses CerbosClient.check_resources() exactly as required by your client code.
    """

    # Principal (roles must be a SET)
    principal = Principal(
        id="user",
        roles={role},
        attr={"role": role},
    )

    # Resource (attr MUST be used, not attributes)
    resource_id = f"invoice_template_{stage}"
    resource = Resource(
        id=resource_id,
        kind="template",
        attr={"stage": stage},
    )

    # Wrap resource inside ResourceAction → inside ResourceList
    res_list = ResourceList(
        resources=[
            ResourceAction(
                resource=resource,
                actions={"promote"},
            )
        ]
    )

    try:
        # Instantiate client locally to avoid long-lived idle gRPC connections timing out
        with CerbosClient(host=CERBOS_HOST) as client:
            resp = client.check_resources(
                principal=principal,
                resources=res_list,
            )

        # If no results → deny (or allow if not strict)
        if not resp or not resp.results:
            return False if CERBOS_STRICT else True

        # Find our result
        result = resp.get_resource(resource_id)
        if not result:
            return False if CERBOS_STRICT else True

        # Check allow/deny
        return result.is_allowed("promote")

    except Exception as e:
        print(f"[CERBOS] Error while checking promote: {e}")
        # Allow fallback if not strict
        return False if CERBOS_STRICT else True
