
from typing import TypedDict, Optional, Dict, List, Any

class InvoiceState(TypedDict, total=False):
    pdf_path: str
    text: str
    page_texts: List[str]
    page_sections: List[Dict[str, Any]]
    images: List[str]
    signature: str
    vendor: str
    template_active: Optional[Dict]
    template_staging: Optional[Dict]
    template_source: str
    template_resolution_mode: str
    template: Optional[Dict]
    suggested_signatures: List[str]
    fields: Dict[str, Any]
    node_timings: Dict[str, float]
    field_extraction_backend: str
    embedding_backend: str
    embedding_model: str
    template_learning_backend: str
    template_learning_model: str
    vision_backend: str
    vision_model: str
    validation_mode: str
    validation_decision: str
    audit_warning: bool
    audit_reason: str
    ollama_runtime: Dict[str, Any]
    vision_pass: bool
    vision_score: float
    vision_critique: str
    promotion_status: str
    role: str
    done: bool
