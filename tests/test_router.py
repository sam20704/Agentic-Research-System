from src.document.models import DocumentProfile
from src.ingestion.router import (
    ParserType,
    RoutingPolicy,
    route_document,
)


def make_profile(**overrides):
    values = {
        "file_path": "test.pdf",
        "page_count": 10,
        "text_coverage": 0.95,
        "avg_chars_per_page": 1500,
        "scanned_page_ratio": 0.0,
        "total_characters": 15000,
        "total_images": 0,
        "images_per_page": 0.0,
        "total_text_blocks": 10,
        "avg_blocks_per_page": 10.0,
        "estimated_table_count": 0,
        "has_complex_layout": False,
        "detected_language": "en",
        "profiling_time_ms": 10.0,
    }

    values.update(overrides)

    return DocumentProfile(**values)


def test_normal_digital_pdf_uses_fast_native():
    profile = make_profile()

    decision = route_document(profile)

    assert decision.parser == ParserType.FAST_NATIVE
    assert decision.rule == "NATIVE_DEFAULT"


def test_scan_heavy_pdf_uses_unlimited_ocr():
    profile = make_profile(
        text_coverage=0.10,
        scanned_page_ratio=0.90,
        total_images=20,
        images_per_page=2.0,
    )

    decision = route_document(profile)

    assert decision.parser == ParserType.UNLIMITED_OCR
    assert decision.rule == "SCAN_HEAVY_AND_LOW_TEXT"


def test_low_text_without_images_does_not_force_ocr():
    profile = make_profile(
        text_coverage=0.10,
        scanned_page_ratio=0.90,
        total_images=0,
    )

    decision = route_document(profile)

    assert decision.parser == ParserType.FAST_NATIVE


def test_mildly_scanned_pdf_stays_native():
    profile = make_profile(
        text_coverage=0.80,
        scanned_page_ratio=0.30,
        total_images=3,
    )

    decision = route_document(profile)

    assert decision.parser == ParserType.FAST_NATIVE


def test_complex_layout_uses_docling():
    profile = make_profile(
        has_complex_layout=True,
        estimated_table_count=3,
    )

    decision = route_document(profile)

    assert decision.parser == ParserType.DOCLING
    assert decision.rule == "COMPLEX_LAYOUT"


def test_complex_layout_overrides_native_default():
    profile = make_profile(
        has_complex_layout=True,
    )

    decision = route_document(profile)

    assert decision.parser == ParserType.DOCLING


def test_strong_ocr_signal_takes_precedence_over_complex_layout():
    profile = make_profile(
        text_coverage=0.10,
        scanned_page_ratio=0.90,
        total_images=10,
        has_complex_layout=True,
    )

    decision = route_document(profile)

    assert decision.parser == ParserType.UNLIMITED_OCR


def test_empty_document_is_rejected():
    profile = make_profile(page_count=0)

    try:
        route_document(profile)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "empty PDF" in str(exc)


def test_decision_contains_explainability_fields():
    profile = make_profile()

    decision = route_document(profile)

    assert decision.reason
    assert decision.rule
    assert 0.0 <= decision.confidence <= 1.0
    assert isinstance(decision.signals, dict)
    assert decision.signals["page_count"] == 10


def test_custom_policy_changes_routing():
    profile = make_profile(
        text_coverage=0.50,
        scanned_page_ratio=0.50,
        total_images=10,
    )

    default_decision = route_document(profile)

    strict_policy = RoutingPolicy(
        scan_heavy_ratio=0.40,
        low_text_coverage=0.60,
    )

    custom_decision = route_document(
        profile,
        policy=strict_policy,
    )

    assert default_decision.parser == ParserType.FAST_NATIVE
    assert custom_decision.parser == ParserType.UNLIMITED_OCR