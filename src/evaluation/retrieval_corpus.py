"""Build the deterministic retrieval benchmark corpus from reference PDFs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.document.models import Document
from src.rag.loader import load_pdf
from src.retrieval.chunking import ChunkingConfig, chunk_document
from src.retrieval.models import DocumentChunk


DEFAULT_REFERENCE_DIR = Path("data/references")


def load_reference_documents(
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
) -> list[Document]:
    """Load all reference PDFs through the canonical PDF loader.

    Files are processed in deterministic filename order.
    """
    reference_path = Path(reference_dir)

    pdf_paths = sorted(
        path
        for path in reference_path.glob("*.pdf")
        if path.is_file()
    )

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {reference_path}"
        )

    return [load_pdf(str(path)) for path in pdf_paths]


def build_retrieval_corpus(
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
    chunking_config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Build canonical DocumentChunk objects from reference PDFs.

    The same resulting chunks must be used by both the BGE-M3
    baseline and the ColBERT benchmark.
    """
    documents = load_reference_documents(reference_dir)

    chunks: list[DocumentChunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                config=chunking_config,
            )
        )

    chunks.sort(key=lambda chunk: chunk.chunk_id)

    _validate_chunks(chunks)

    return chunks


def _validate_chunks(chunks: Sequence[DocumentChunk]) -> None:
    """Validate deterministic and provenance requirements."""
    if not chunks:
        raise ValueError("Retrieval corpus contains no chunks.")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk_id values detected.")

    for chunk in chunks:
        if not chunk.document_id:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has no document_id."
            )

        if not chunk.text.strip():
            raise ValueError(
                f"Chunk {chunk.chunk_id} contains empty text."
            )

        if not chunk.source:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has no source."
            )

        if not chunk.page_numbers:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has no page provenance."
            )


__all__ = [
    "DEFAULT_REFERENCE_DIR",
    "load_reference_documents",
    "build_retrieval_corpus",
]