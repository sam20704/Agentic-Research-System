from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.document.models import DocumentProfile


class ParserType(str, Enum):
    FAST_NATIVE = "fast_native"
    DOCLING = "docling"
    UNLIMITED_OCR = "unlimited_ocr"


@dataclass(frozen=True)
class RoutingPolicy:
    """
    Simple rule-based policy for selecting the document parser.

    OCR takes precedence when the document is clearly scan-heavy.
    Docling is used for structurally complex documents.
    PyMuPDF remains the default fast path.
    """

    scan_heavy_ratio: float = 0.60
    low_text_coverage: float = 0.40


@dataclass
class RoutingDecision:
    """Explainable result of document parser routing."""

    parser: ParserType
    reason: str
    confidence: float
    rule: str
    signals: dict[str, Any] = field(default_factory=dict)


def route_document(
    profile: DocumentProfile,
    policy: RoutingPolicy | None = None,
) -> RoutingDecision:
    """
    Select the appropriate parser using the DocumentProfile.

    Routing order:

        1. Clearly scan-heavy + low native text + images
           -> Unlimited-OCR

        2. Complex layout/table structure
           -> Docling

        3. Everything else
           -> PyMuPDF
    """

    policy = policy or RoutingPolicy()

    if profile.page_count <= 0:
        raise ValueError("Cannot route an empty PDF")

    signals = {
        "page_count": profile.page_count,
        "text_coverage": profile.text_coverage,
        "scanned_page_ratio": profile.scanned_page_ratio,
        "total_images": profile.total_images,
        "images_per_page": profile.images_per_page,
        "has_complex_layout": profile.has_complex_layout,
        "estimated_table_count": profile.estimated_table_count,
    }

    # Rule 1: strong OCR signal.
    if (
        profile.scanned_page_ratio >= policy.scan_heavy_ratio
        and profile.text_coverage <= policy.low_text_coverage
        and profile.total_images > 0
    ):
        return RoutingDecision(
            parser=ParserType.UNLIMITED_OCR,
            reason=(
                "Scan-heavy document with insufficient native text "
                "and image content"
            ),
            confidence=0.90,
            rule="SCAN_HEAVY_AND_LOW_TEXT",
            signals=signals,
        )

    # Rule 2: structurally complex digital document.
    if profile.has_complex_layout:
        return RoutingDecision(
            parser=ParserType.DOCLING,
            reason="Complex document layout or table structure detected",
            confidence=0.85,
            rule="COMPLEX_LAYOUT",
            signals=signals,
        )

    # Rule 3: fast/default native extraction.
    return RoutingDecision(
        parser=ParserType.FAST_NATIVE,
        reason="No strong OCR or structural complexity signals",
        confidence=0.90,
        rule="NATIVE_DEFAULT",
        signals=signals,
    )