from pathlib import Path

from src.rag.loader import load_pdf, load_pdfs


def test_load_pdf():
    pdf_files = sorted(Path("data/references").glob("*.pdf"))

    assert pdf_files, "No reference PDFs found."

    document = load_pdf(str(pdf_files[0]))

    assert document.page_count > 0
    assert document.source
    assert document.file_hash
    assert document.pages


def test_load_pdfs_legacy_compatibility():
    documents = load_pdfs("data/references")

    assert documents

    document = documents[0]

    assert document["source"]
    assert document["content"]
    assert document["document_id"]
    assert document["file_hash"]
    assert document["page_count"] > 0
