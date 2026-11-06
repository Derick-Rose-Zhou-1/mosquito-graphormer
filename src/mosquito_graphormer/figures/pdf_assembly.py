"""Vector PDF sizing and Fig. 5 assembly from external, author-owned artwork."""
import argparse
from pathlib import Path
import fitz


def single_page(path):
    doc = fitz.open(path)
    if len(doc) != 1:
        doc.close()
        raise ValueError(f"Expected one PDF page: {path}")
    return doc


def assemble(output, statistical, artwork=None):
    """Size Fig. 2, or assemble the final Fig. 5 layout without rasterization."""
    output = Path(output)
    if output.resolve() in {Path(p).resolve() for p in (statistical, artwork) if p}:
        raise ValueError("Output must not overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    with single_page(statistical) as source, fitz.open() as result:
        if artwork is None:
            page = result.new_page(width=510.236206, height=381.580688)
            page.show_pdf_page(page.rect, source, 0, keep_proportion=False)
        else:
            page = result.new_page(width=518.740173, height=420.586884)
            with single_page(artwork) as diagram:
                page.show_pdf_page(fitz.Rect(0, 0, page.rect.width, 166.748719), diagram, 0,
                                   keep_proportion=False)
            page.show_pdf_page(fitz.Rect(0, 172.748719, page.rect.width, page.rect.height),
                               source, 0, keep_proportion=False)
        result.save(output, garbage=4, deflate=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statistical", type=Path, required=True)
    p.add_argument("--artwork", type=Path, help="Fig. 5a PDF; omit when sizing Fig. 2")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assemble(args.output, args.statistical, args.artwork)


if __name__ == "__main__":
    main()
