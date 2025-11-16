
# Local Secure Invoice Extractor (LangGraph + Milvus + Cerbos)

Reads invoice PDFs → extracts totals and tax → reuses cached templates → learns new templates (staging) → validates with LLaVA → promotes via Cerbos.

## Quickstart
```bash
docker-compose up -d
make models

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

APP_ROLE=manager python -m src.graph_invoice.run_invoice_graph samples/invoices/invoice1.pdf

$ python -m src.invoice.templates_cli list
Active:

Staging:
   acme_corporation_123_main_street_invoice_72246d14


$ python -m src.invoice.templates_cli reject acme_corporation_123_main_street_invoice_72246d14
Rejected (deleted from staging).

```
System packages for OCR: `sudo apt-get install -y tesseract-ocr poppler-utils`
