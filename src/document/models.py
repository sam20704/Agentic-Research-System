from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    """
    Page-relative bounding box.

    Coordinates follow the source parser's coordinate system:
    x0, y0 = top-left
    x1, y1 = bottom-right
    """
    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }


@dataclass
class Element:
    """
    Atomic document element such as text, title, table, figure, etc.
    """
    element_id: str
    element_type: str
    text: str = ""
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "text": self.text,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "metadata": self.metadata,
        }


@dataclass
class Page:
    """
    Canonical representation of one source PDF page.
    """
    page_number: int
    text: str = ""
    width: float | None = None
    height: float | None = None
    elements: list[Element] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "width": self.width,
            "height": self.height,
            "elements": [element.to_dict() for element in self.elements],
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """
    Canonical document representation shared by all ingestion backends.
    """
    document_id: str
    source: str
    source_path: str
    file_hash: str
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(
            page.text.strip()
            for page in self.pages
            if page.text.strip()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "source_path": self.source_path,
            "file_hash": self.file_hash,
            "page_count": self.page_count,
            "pages": [page.to_dict() for page in self.pages],
            "metadata": self.metadata,
        }


@dataclass
class DocumentProfile:
    """
    Diagnostic profile and metric report card for an ingested PDF document.
    """
    file_path: str
    page_count: int
    text_coverage: float
    avg_chars_per_page: float
    scanned_page_ratio: float
    total_characters: int = 0
    total_images: int = 0
    images_per_page: float = 0.0
    total_text_blocks: int = 0
    avg_blocks_per_page: float = 0.0
    estimated_table_count: int = 0
    has_complex_layout: bool = False
    detected_language: str = "unknown"
    profiling_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "page_count": self.page_count,
            "text_coverage": round(self.text_coverage, 4),
            "avg_chars_per_page": round(self.avg_chars_per_page, 2),
            "scanned_page_ratio": round(self.scanned_page_ratio, 4),
            "total_characters": self.total_characters,
            "total_images": self.total_images,
            "images_per_page": round(self.images_per_page, 2),
            "total_text_blocks": self.total_text_blocks,
            "avg_blocks_per_page": round(self.avg_blocks_per_page, 2),
            "estimated_table_count": self.estimated_table_count,
            "has_complex_layout": self.has_complex_layout,
            "detected_language": self.detected_language,
            "profiling_time_ms": round(self.profiling_time_ms, 2),
            "metadata": self.metadata,
        }

