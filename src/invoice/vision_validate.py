import json
import os
import tempfile
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Optional

from PIL import Image

from src.invoice.ollama_runtime import VALIDATION_MODEL, VISION_MODEL, get_ollama_llm

VALIDATOR = None
VALIDATION_TIMEOUT_SECONDS = float(os.getenv("INVOICE_VISION_TIMEOUT_SECONDS", "300"))
VALIDATION_USE_IMAGES = os.getenv("INVOICE_VALIDATION_USE_IMAGES", "false").lower() == "true"
VISION_MAX_DIM = int(os.getenv("INVOICE_VISION_MAX_DIM", "1600"))
VISION_JPEG_QUALITY = int(os.getenv("INVOICE_VISION_JPEG_QUALITY", "70"))
VISION_MAX_PAGES = int(os.getenv("INVOICE_VISION_MAX_PAGES", "2"))
VISION_HEADER_RATIO = float(os.getenv("INVOICE_VISION_HEADER_RATIO", "0.35"))
VISION_SUMMARY_START_RATIO = float(os.getenv("INVOICE_VISION_SUMMARY_START_RATIO", "0.55"))


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


def _extract_first_json_object(text: str) -> Optional[Dict]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        ch = text[index]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _prepare_vision_images(images: List[str]) -> List[str]:
    prepared: List[str] = []
    digest = hashlib.sha256("|".join(images).encode("utf-8")).hexdigest()[:12]
    out_dir = os.path.join(tempfile.gettempdir(), f"invoice-vision-{digest}")
    os.makedirs(out_dir, exist_ok=True)

    for index, image_path in enumerate(images[:VISION_MAX_PAGES]):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size

                header = img.crop((0, 0, width, max(1, int(height * VISION_HEADER_RATIO))))
                header.thumbnail((VISION_MAX_DIM, VISION_MAX_DIM))
                header_path = os.path.join(out_dir, f"page-{index + 1}-header.jpg")
                header.save(header_path, format="JPEG", quality=VISION_JPEG_QUALITY, optimize=True)
                prepared.append(header_path)

                summary_top = min(height - 1, max(0, int(height * VISION_SUMMARY_START_RATIO)))
                summary = img.crop((0, summary_top, width, height))
                summary.thumbnail((VISION_MAX_DIM, VISION_MAX_DIM))
                summary_path = os.path.join(out_dir, f"page-{index + 1}-summary.jpg")
                summary.save(summary_path, format="JPEG", quality=VISION_JPEG_QUALITY, optimize=True)
                prepared.append(summary_path)
        except Exception:
            prepared.append(image_path)

    return prepared or images[:VISION_MAX_PAGES]


def validate_with_vision(
    answer_fields: Dict,
    images: List[str],
    evidence: Optional[Dict] = None,
) -> Dict:
    prepared_images = _prepare_vision_images(images) if (images and VALIDATION_USE_IMAGES) else []
    validation_model = VISION_MODEL if prepared_images else VALIDATION_MODEL

    prompt = (
        "You are an expert invoice auditor. Validate the extracted invoice fields against the structured evidence JSON. "
        "Treat numerically equivalent values as matching even if formatting differs, for example '$107.50' vs '107,50 kr'. "
        "Treat the presence or absence of 'kr', '$', comma decimals, period decimals, whitespace, and localized field labels as acceptable normalization differences, not mismatches. "
        "Treat equivalent date formats as matching if they represent the same calendar date, for example '2025-11-06', '2025/11/06', '06-11-2025', or '06.11.2025'. "
        "Do NOT complain about formatting-only differences if the numeric value and meaning are the same. "
        "Only report a mismatch when the actual value or semantic meaning differs, or when the evidence JSON does not support the extracted field. "
        "The evidence JSON may already contain normalized lines and candidate totals from the PDF text. "
        "If optional images are provided, use them only as secondary support, not as the primary source.\n\n"
        "If the extracted fields are supported by the evidence JSON, return pass=true with a short critique. "
        "If evidence is insufficient, say that explicitly instead of inventing discrepancies.\n\n"
        "Return ONLY a single JSON object in this exact format:\n"
        '{"score": float, "pass": bool, "critique": string}\n\n'
        f"EXTRACTED_FIELDS_JSON:\n{json.dumps(answer_fields, indent=2, ensure_ascii=False)}\n\n"
        f"EVIDENCE_JSON:\n{json.dumps(evidence or {}, indent=2, ensure_ascii=False)}\n"
    )

    try:
        global VALIDATOR
        if VALIDATOR is None:
            VALIDATOR = get_ollama_llm(validation_model, num_ctx=16384)
        executor = ThreadPoolExecutor(max_workers=1)
        kwargs = {"images": prepared_images} if prepared_images else {}
        future = executor.submit(VALIDATOR.invoke, prompt, **kwargs)
        try:
            raw = future.result(timeout=VALIDATION_TIMEOUT_SECONDS)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    except FuturesTimeoutError:
        future.cancel()
        return {
            "pass": False,
            "score": 0.0,
            "critique": f"Validation model timed out after {VALIDATION_TIMEOUT_SECONDS:.0f}s",
            "backend": "timeout",
            "model": validation_model,
            "images_sent": len(prepared_images),
        }
    except Exception as exc:
        return {
            "pass": False,
            "score": 0.0,
            "critique": f"Validation model unavailable: {exc}",
            "backend": "unavailable",
            "model": validation_model,
            "images_sent": len(prepared_images),
        }

    text = _strip_code_fence(raw)
    try:
        obj = json.loads(text)
    except Exception:
        obj = _extract_first_json_object(text)
        if obj is None:
            return {
                "pass": False,
                "score": 0.5,
                "critique": raw[:500],
                "backend": "ollama_unparsed",
                "model": validation_model,
                "images_sent": len(prepared_images),
            }

    return {
        "score": float(obj.get("score", 0.0)),
        "pass": bool(obj.get("pass", False)),
        "critique": str(obj.get("critique", "")),
        "backend": "ollama_json" if not prepared_images else "ollama_multimodal",
        "model": validation_model,
        "images_sent": len(prepared_images),
    }
