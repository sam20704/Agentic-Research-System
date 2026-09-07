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
