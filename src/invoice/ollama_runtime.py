import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
TEXT_MODEL = os.getenv("INVOICE_TEMPLATE_LLM_MODEL", "llama3.2")
VISION_MODEL = os.getenv("INVOICE_VISION_MODEL", "qwen3-vl:8b")
EMBED_MODEL = os.getenv("INVOICE_EMBED_MODEL", "nomic-embed-text")
VALIDATION_MODEL = os.getenv("INVOICE_VALIDATION_MODEL", TEXT_MODEL)

_status_cache: Optional[Dict[str, Any]] = None
_llm_cache: Dict[tuple[str, int | None], OllamaLLM] = {}
_embed_cache: Dict[str, OllamaEmbeddings] = {}


def get_ollama_status(force_refresh: bool = False) -> Dict[str, Any]:
    global _status_cache
    if _status_cache is not None and not force_refresh:
        return _status_cache

    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    status = {
        "available": False,
        "base_url": OLLAMA_BASE_URL,
        "models": [],
        "error": None,
    }
    try:
        with urllib.request.urlopen(endpoint, timeout=2.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        status["available"] = True
        status["models"] = [m.get("name") for m in payload.get("models", []) if m.get("name")]
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        status["error"] = str(exc)

    _status_cache = status
    return status


def get_ollama_llm(model: str, *, num_ctx: int | None = None) -> OllamaLLM:
    key = (model, num_ctx)
    llm = _llm_cache.get(key)
    if llm is None:
        kwargs: Dict[str, Any] = {"model": model}
        if num_ctx is not None:
            kwargs["num_ctx"] = num_ctx
        llm = OllamaLLM(**kwargs)
        _llm_cache[key] = llm
    return llm


def get_ollama_embeddings(model: str) -> OllamaEmbeddings:
    embedder = _embed_cache.get(model)
    if embedder is None:
        embedder = OllamaEmbeddings(model=model)
        _embed_cache[model] = embedder
    return embedder


def summarize_ollama_runtime() -> Dict[str, Any]:
    status = get_ollama_status()
    return {
        "available": status["available"],
        "base_url": status["base_url"],
        "models": status["models"],
        "text_model": TEXT_MODEL,
        "validation_model": VALIDATION_MODEL,
        "vision_model": VISION_MODEL,
        "embed_model": EMBED_MODEL,
        "error": status["error"],
    }
