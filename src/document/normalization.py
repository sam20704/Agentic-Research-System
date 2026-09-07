from .models import Document


def document_to_text(document: Document) -> str:
    """
    Convert a canonical document back into plain text.

    This provides compatibility with the current chunking pipeline
    while preserving the canonical representation internally.
    """
    return document.text


def document_to_legacy_dict(document: Document) -> dict:
    """
    Compatibility adapter for the current RAG pipeline.
    """
    return {
        "source": document.source,
        "content": document.text,
        "document_id": document.document_id,
        "source_path": document.source_path,
        "file_hash": document.file_hash,
        "page_count": document.page_count,
    }
