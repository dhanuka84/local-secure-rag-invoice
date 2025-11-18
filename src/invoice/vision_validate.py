
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

    prompt = (
        "You are verifying extracted invoice amounts against images. "
        "Given the extracted fields in JSON, check if the images show the same "
        "subtotal, tax, and total. "
        "Return ONLY JSON with: score (0..1), pass (true/false), critique.\n\n"
        f"EXTRACTED_JSON:\n{json.dumps(answer_fields)}\n"
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
