
from typing import TypedDict, Optional, Dict, List, Any

class InvoiceState(TypedDict, total=False):
    pdf_path: str
    text: str
    images: List[str]
    signature: str
    vendor: str
    template_active: Optional[Dict]
    template_staging: Optional[Dict]
    template_source: str
    template: Optional[Dict]
    suggested_signatures: List[str]
    fields: Dict[str, Any]
    vision_pass: bool
    vision_score: float
    vision_critique: str
    promotion_status: str
    role: str
    done: bool
