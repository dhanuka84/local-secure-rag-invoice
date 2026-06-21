
import hashlib
import os
import tempfile
from typing import Dict, List, Tuple
from PIL import Image
import pdfplumber

def pdf_to_text_and_images(pdf_path: str) -> Tuple[str, List[str]]:
    texts, images = [], []
    pdf_name = os.path.basename(pdf_path)
    pdf_hash = hashlib.sha256(pdf_path.encode("utf-8")).hexdigest()[:12]
    image_dir = os.path.join(tempfile.gettempdir(), f"invoice-pages-{pdf_hash}")
    os.makedirs(image_dir, exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            texts.append(txt)
            pil = page.to_image(resolution=300).original
            img_path = os.path.join(image_dir, f"{pdf_name}.page{i+1}.png")
            pil.save(img_path)
            images.append(img_path)
    full_text = "\n".join(texts).strip()
    return full_text, images


def pdf_to_page_bundle(pdf_path: str) -> Dict[str, List[str] | str]:
    text, images = pdf_to_text_and_images(pdf_path)
    page_texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_texts.append((page.extract_text() or "").strip())
    return {
        "text": text,
        "images": images,
        "page_texts": page_texts,
    }

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
