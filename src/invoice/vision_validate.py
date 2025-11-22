
import json
from typing import List, Dict
from langchain_ollama import OllamaLLM

#VLM = OllamaLLM(model="llava")
VLM = OllamaLLM(model="qwen3-vl:8b")

def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def validate_with_vision(answer_fields: Dict, images: List[str]) -> Dict:
    if not images:
        return {"pass": True, "score": 1.0, "critique": "No images to validate."}

    # 1. Prepare Key Metrics for Direct Check
    subtotal = answer_fields.get("subtotal", "N/A")
    tax = answer_fields.get("tax", "N/A")
    total = answer_fields.get("total_amount", "N/A") # Assuming total_amount is the key
    invoice_date = answer_fields.get("invoice_date", "N/A")
    
    # 2. Construct the Detailed Prompt
    prompt = (
        "You are an expert invoice auditor. Your task is to verify the extracted data "
        "against the provided invoice image. The invoice is in Swedish (kr). "
        "Verify the following key amounts and data points by locating them on the image: \n"
        f"1. **Invoice Total (Totalt belopp)**: MUST match **{total} kr** \n"
        f"2. **Subtotal (Summa exkl moms)**: MUST match **{subtotal} kr** \n"
        f"3. **Tax (Moms)**: MUST match **{tax} kr** \n"
        f"4. **Invoice Date (Fakturadatum)**: MUST match **{invoice_date}** \n\n"
        
        "CRITIQUE GUIDELINES:\n"
        "Critique why the score is NOT 1.0 (e.g., 'Total amount is 172,00 kr in the image, but extracted 172.0 kr').\n"
        
        "Return ONLY a single, clean JSON object that strictly adheres to the format:\n"
        '{"score": float, "pass": bool, "critique": string}\n\n'
        f"EXTRACTED_JSON for cross-reference:\n{json.dumps(answer_fields, indent=2)}\n"
    )

    raw = VLM.invoke(prompt, images=images)
    text = _strip_code_fence(raw)

    try:
        obj = json.loads(text)
        score = float(obj.get("score", 0.0))
        passed = bool(obj.get("pass", False))
        critique = str(obj.get("critique", ""))
        return {"score": score, "pass": passed, "critique": critique}
    except Exception:
        return {"pass": False, "score": 0.5, "critique": raw[:500]}
