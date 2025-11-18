
.PHONY: models run test templates-list

models:
	-ollama pull llama3.2 || true
	-ollama pull nomic-embed-text || true
	#-ollama pull llava || true
	-ollama pull qwen3-vl:8b || true

run:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	APP_ROLE=manager python -m src.graph_invoice.run_invoice_graph samples/invoices/invoice1.pdf

templates-list:
	. .venv/bin/activate && python -m src.invoice.templates_cli list

test:
	. .venv/bin/activate && pip install -r requirements-dev.txt && pytest -q
