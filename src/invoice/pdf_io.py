
import io
from typing import Tuple, List
from PIL import Image
import pdfplumber

def pdf_to_text_and_images(pdf_path: str) -> Tuple[str, List[str]]:
    texts, images = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            texts.append(txt)
            pil = page.to_image(resolution=300).original
            img_path = f"{pdf_path}.page{i+1}.png"
            pil.save(img_path)
            images.append(img_path)
    full_text = "\n".join(texts).strip()
    return full_text, images

def ocr_if_needed(text: str, image_paths: List[str]) -> str:
    if text.strip():
        return text
    try:
        import pytesseract
    except Exception:
        return text
    extracted = []
    for p in image_paths:
        try:
            img = Image.open(p)
            extracted.append(pytesseract.image_to_string(img))
        except Exception:
            pass
    return "\n".join(extracted).strip() or text
