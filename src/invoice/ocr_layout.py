from typing import Any, Dict, List
from pathlib import Path

from PIL import Image
from .pdf_io import pdf_to_text_and_images

try:
    import pytesseract
except Exception:  # optional dependency
    pytesseract = None


def ocr_with_layout(pdf_path: str) -> Dict[str, Any]:
    # Run OCR with bounding boxes using Tesseract on PDF-rendered images.
    text, image_paths = pdf_to_text_and_images(str(Path(pdf_path)))
    pages: List[Dict[str, Any]] = []

    if not pytesseract or not image_paths:
        return {"text": text, "pages": pages}

    for page_index, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)
        except Exception:
            continue

        width, height = img.size

        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except Exception:
            pages.append(
                {
                    "page_index": page_index,
                    "width": width,
                    "height": height,
                    "tokens": [],
                }
            )
            continue

        tokens: List[Dict[str, Any]] = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            token = {
                "text": txt,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
            tokens.append(token)

        pages.append(
            {
                "page_index": page_index,
                "width": width,
                "height": height,
                "tokens": tokens,
            }
        )

    return {"text": text, "pages": pages}
