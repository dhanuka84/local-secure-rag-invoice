import argparse
from pathlib import Path

from PIL import Image
import pdfplumber


def downscale_pdf(input_pdf: str, output_pdf: str, *, dpi: int, max_dim: int, jpeg_quality: int):
    processed = []
    with pdfplumber.open(input_pdf) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=dpi).original.convert("RGB")
            img.thumbnail((max_dim, max_dim))
            processed.append(img)

    if not processed:
        raise ValueError(f"No pages found in PDF: {input_pdf}")

    first, rest = processed[0], processed[1:]
    first.save(
        output_pdf,
        format="PDF",
        save_all=True,
        append_images=rest,
        resolution=dpi,
        quality=jpeg_quality,
    )


def main():
    parser = argparse.ArgumentParser(description="Create a downscaled PDF for faster local vision processing.")
    parser.add_argument("input_pdf")
    parser.add_argument("output_pdf")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-dim", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    args = parser.parse_args()

    output_path = Path(args.output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downscale_pdf(
        args.input_pdf,
        str(output_path),
        dpi=args.dpi,
        max_dim=args.max_dim,
        jpeg_quality=args.jpeg_quality,
    )
    print(output_path)


if __name__ == "__main__":
    main()
