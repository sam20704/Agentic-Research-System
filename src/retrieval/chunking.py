"""Chunk canonical documents into retrieval-ready units."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.document.models import Document, Element
from src.retrieval.models import DocumentChunk


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = 1200
    chunk_overlap: int = 180
    min_chunk_chars: int = 80

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        if self.min_chunk_chars <= 0:
            raise ValueError("min_chunk_chars must be greater than 0")


def _make_chunk_id(
    document_id: str,
    page_numbers: tuple[int, ...],
    chunk_index: int,
    text: str,
) -> str:
    """Create a deterministic chunk identifier."""

    payload = (
        f"{document_id}|"
        f"{','.join(map(str, page_numbers))}|"
        f"{chunk_index}|"
        f"{text}"
    )

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    return f"{document_id}:chunk:{chunk_index}:{digest}"


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text into overlapping character-based chunks.

    The splitter prefers paragraph and whitespace boundaries while
    guaranteeing progress for long individual paragraphs.
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = text.rfind("\n\n", start, end)

            if boundary <= start:
                boundary = text.rfind("\n", start, end)

            if boundary <= start:
                boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - chunk_overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def _element_text(element: Element) -> str:
    """Return normalized element text."""

    return " ".join(element.text.split())


def chunk_document(
    document: Document,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Convert a canonical Document into retrieval chunks.

    Element provenance is preserved whenever chunks are generated
    directly from canonical document elements.
    """

    config = config or ChunkingConfig()

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in document.pages:
        page_elements = [
            element
            for element in page.elements
            if element.text and element.text.strip()
        ]

        if page_elements:
            for element in page_elements:
                text = _element_text(element)

                if len(text) < config.min_chunk_chars:
                    continue

                pieces = _split_text(
                    text=text,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )

                for piece in pieces:
                    if len(piece) < config.min_chunk_chars:
                        continue

                    page_numbers = (page.page_number,)

                    chunk_id = _make_chunk_id(
                        document_id=document.document_id,
                        page_numbers=page_numbers,
                        chunk_index=chunk_index,
                        text=piece,
                    )

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document.document_id,
                            text=piece,
                            page_numbers=page_numbers,
                            source=document.source,
                            bounding_boxes=(element.bbox,),
                            element_ids=(element.element_id,),
                            metadata={
                                "parser": document.metadata.get(
                                    "parser"
                                ),
                                "chunk_index": chunk_index,
                                "element_type": element.element_type,
                            },
                        )
                    )

                    chunk_index += 1

        elif page.text.strip():
            text = " ".join(page.text.split())

            pieces = _split_text(
                text=text,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            for piece in pieces:
                if len(piece) < config.min_chunk_chars:
                    continue

                page_numbers = (page.page_number,)

                chunk_id = _make_chunk_id(
                    document_id=document.document_id,
                    page_numbers=page_numbers,
                    chunk_index=chunk_index,
                    text=piece,
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        text=piece,
                        page_numbers=page_numbers,
                        source=document.source,
                        metadata={
                            "parser": document.metadata.get("parser"),
                            "chunk_index": chunk_index,
                        },
                    )
                )

                chunk_index += 1

    return chunks