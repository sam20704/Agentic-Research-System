from pathlib import Path

from src.rag.loader import load_pdf


def test_canonical_pdf_document():
    pdf_files = sorted(Path("data/references").glob("*.pdf"))

    assert pdf_files, "No reference PDFs found."

    document = load_pdf(str(pdf_files[0]))

    assert document.document_id.startswith("doc_")
    assert document.source
    assert document.source_path
    assert document.file_hash
    assert len(document.file_hash) == 64

    assert document.page_count > 0

    page = document.pages[0]

    assert page.page_number == 1
    assert page.width is not None
    assert page.height is not None

    if page.elements:
        element = page.elements[0]

        assert element.element_id
        assert element.element_type == "text"

        if element.bbox:
            assert element.bbox.x1 >= element.bbox.x0
            assert element.bbox.y1 >= element.bbox.y0
