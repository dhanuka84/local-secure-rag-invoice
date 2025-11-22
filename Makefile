.PHONY: models run test templates-list install-deps setup

# ------------------------------
# Models (Ollama pulls)
# ------------------------------
models:
	-ollama pull llama3.2 || true
	-ollama pull nomic-embed-text || true
	-ollama pull qwen3-vl:8b || true

# ------------------------------
# Smart Dependency Installer
# ------------------------------

APT_PKGS = tesseract-ocr tesseract-ocr-swe libleptonica-dev libtesseract-dev poppler-utils

install-deps:
	@echo "==> Checking APT packages..."
	@missing=""; \
	for p in $(APT_PKGS); do \
		if ! dpkg -s $$p >/dev/null 2>&1; then \
			missing="$$missing $$p"; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "==> Installing missing APT packages:$$missing"; \
		sudo apt update && sudo apt install -y $$missing; \
	else \
		echo "==> All APT packages already installed"; \
	fi

	@echo "==> Checking spaCy..."
	@python3 -c "import spacy" >/dev/null 2>&1 || \
		( echo "   -> Installing spaCy"; pip install spacy )

	@echo "==> Checking Swedish language model (sv_core_news_lg)..."
	@python3 -c "import sv_core_news_lg" >/dev/null 2>&1 || \
		( echo "   -> Downloading sv_core_news_lg"; python3 -m spacy download sv_core_news_lg )

# ------------------------------
# Run Pipeline
# ------------------------------
run:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	APP_ROLE=manager python -m src.graph_invoice.run_invoice_graph samples/invoices/invoice1.pdf

# ------------------------------
# Template Management
# ------------------------------
templates-list:
	. .venv/bin/activate && python -m src.invoice.templates_cli list

# ------------------------------
# Tests
# ------------------------------
test:
	. .venv/bin/activate && pip install -r requirements-dev.txt && pytest -q

# ------------------------------
# One-shot Setup (Optional)
# ------------------------------
setup: install-deps models
	@echo "==> Setup complete!"
