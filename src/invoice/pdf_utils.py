from typing import List
from PIL import Image
from pdf2image import convert_from_path

def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Render a PDF to a list of PIL.Image objects (one per page)."""
    return convert_from_path(pdf_path)
