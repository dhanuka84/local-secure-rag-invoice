from src.invoice.extract import extract_fields
from src.invoice.pdf_io import pdf_to_text_and_images


def _extract_from_pdf(pdf_path: str) -> dict:
    text, _ = pdf_to_text_and_images(pdf_path)
    return extract_fields(text, {"regex": {}})


def test_extracts_english_invoice_fields():
    fields = _extract_from_pdf("samples/invoices/invoice1.pdf")
    print("invoice1.pdf", fields)

    assert fields == {
        "invoice_no": "INV-1001",
        "date": "2025-11-05",
        "subtotal": "100.00",
        "tax": "7.50",
        "total": "107.50",
        "tax_rate": "0.0750",
    }


def test_extracts_swedish_invoice_fields():
    fields = _extract_from_pdf("samples/invoices/godel.pdf")
    print("godel.pdf", fields)

    assert fields == {
        "invoice_no": "2687252805",
        "date": "2025-11-06",
        "subtotal": "137.25",
        "tax": "34.32",
        "total": "172.00",
        "tax_rate": "0.25",
    }
