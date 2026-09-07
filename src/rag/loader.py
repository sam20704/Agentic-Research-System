import os

import fitz

from src.document import (
    BoundingBox,
    Document,
    Element,
    Page,
    calculate_file_hash,
    make_document_id,
    make_element_id,
)


def load_pdf(file_path: str) -> Document:
    """
    Load one PDF into the canonical Document representation.

    PyMuPDF remains the fast/native extraction path for now.
    More advanced parsers will later implement the same canonical model.
    """
    file_hash = calculate_file_hash(file_path)
    document_id = make_document_id(file_hash)

    source = os.path.basename(file_path)

    pdf = fitz.open(file_path)

    pages = []

    try:
        for page_index, pdf_page in enumerate(pdf):
            page_number = page_index + 1

            text = pdf_page.get_text("text").strip()

            rect = pdf_page.rect

            elements = []

            # Preserve text blocks and their page coordinates.
            blocks = pdf_page.get_text("blocks")

            for element_index, block in enumerate(blocks):
                if len(block) < 5:
                    continue

                x0, y0, x1, y1, block_text = block[:5]

                block_text = str(block_text).strip()

                if not block_text:
                    continue

                element = Element(
                    element_id=make_element_id(
                        document_id,
                        page_number,
                        element_index,
                    ),
                    element_type="text",
                    text=block_text,
                    bbox=BoundingBox(
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                    ),
                    metadata={
                        "parser": "pymupdf",
                    },
                )

                elements.append(element)

            pages.append(
                Page(
                    page_number=page_number,
                    text=text,
                    width=float(rect.width),
                    height=float(rect.height),
                    elements=elements,
                    metadata={
                        "parser": "pymupdf",
                    },
                )
            )

    finally:
        pdf.close()

    return Document(
        document_id=document_id,
        source=source,
        source_path=os.path.abspath(file_path),
        file_hash=file_hash,
        pages=pages,
        metadata={
            "parser": "pymupdf",
        },
    )


def load_pdfs(folder_path: str) -> list[dict]:
    """
    Compatibility wrapper for the existing RAG pipeline.

    The canonical Document is created first, then converted to the
    legacy dictionary shape currently consumed by the pipeline.
    """
    from src.document.normalization import document_to_legacy_dict

    documents = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, filename)

        document = load_pdf(file_path)

        documents.append(
            document_to_legacy_dict(document)
        )

    return documents
