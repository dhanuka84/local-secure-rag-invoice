import json

from src.invoice.ollama_runtime import (
    TEXT_MODEL,
    get_ollama_llm,
    summarize_ollama_runtime,
)


def main():
    runtime = summarize_ollama_runtime()
    print(json.dumps(runtime, indent=2))

    if not runtime.get("available"):
        return

    llm = get_ollama_llm(TEXT_MODEL)
    prompt = 'Return only this JSON: {"ok": true}'
    raw = llm.invoke(prompt)
    print(raw)


if __name__ == "__main__":
    main()
