from pathlib import Path
import pytest
import pymupdf

from src.ingestion.profile import profile_pdf
from src.document.models import DocumentProfile


def test_profile_normal_digital_pdf():
    pdf_files = sorted(Path("data/references").glob("*.pdf"))
    assert pdf_files, "No reference PDFs found in data/references."

    target_pdf = next(
        (f for f in pdf_files if "taiwan" in f.name.lower()),
        pdf_files[0]
    )

    profile = profile_pdf(str(target_pdf))

    assert isinstance(profile, DocumentProfile)
    assert profile.page_count > 0
    assert profile.text_coverage >= 0.85
    assert profile.scanned_page_ratio <= 0.15
    assert profile.avg_chars_per_page > 500
    assert profile.total_characters > 0
    assert profile.detected_language == "en"
    assert profile.profiling_time_ms > 0
    assert profile.metadata["meaningful_text_pages"] > 0


def test_profile_scanned_pdf(tmp_path):
    scanned_pdf_path = tmp_path / "synthetic_scanned.pdf"

    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), 1)
    pix.clear_with(255)

    p1 = doc.new_page(width=400, height=400)
    p1.insert_image(p1.rect, pixmap=pix)

    p2 = doc.new_page(width=400, height=400)
    p2.insert_image(p2.rect, pixmap=pix)

    doc.save(str(scanned_pdf_path))
    doc.close()

    profile = profile_pdf(str(scanned_pdf_path))

    assert profile.page_count == 2
    assert profile.text_coverage == 0.0
    assert profile.scanned_page_ratio == 1.0
    assert profile.total_characters == 0
    assert profile.avg_chars_per_page == 0.0
    assert profile.total_images == 2
    assert profile.images_per_page == 1.0
    assert profile.detected_language == "unknown"


def test_profile_to_dict():
    pdf_files = sorted(Path("data/references").glob("*.pdf"))
    profile = profile_pdf(str(pdf_files[0]))
    profile_dict = profile.to_dict()

    expected_keys = {
        "file_path",
        "page_count",
        "text_coverage",
        "avg_chars_per_page",
        "scanned_page_ratio",
        "total_characters",
        "total_images",
        "images_per_page",
        "total_text_blocks",
        "avg_blocks_per_page",
        "estimated_table_count",
        "has_complex_layout",
        "detected_language",
        "profiling_time_ms",
        "metadata"
    }

    assert expected_keys.issubset(profile_dict.keys())
    assert isinstance(profile_dict["page_count"], int)
    assert isinstance(profile_dict["text_coverage"], float)
    assert isinstance(profile_dict["profiling_time_ms"], float)


def test_profile_real_corpus_scanned_pdf():
    real_scanned_path = Path("data/references/mohi_fame-2_book_design.pdf")
    if not real_scanned_path.exists():
        pytest.skip("Real corpus scanned PDF not found")

    profile = profile_pdf(str(real_scanned_path))

    assert profile.page_count == 21
    assert profile.text_coverage == 0.0
    assert profile.scanned_page_ratio == 1.0
    assert profile.total_characters == 0
    assert profile.total_images > 0
    assert profile.images_per_page > 5.0


def test_profile_non_existent_file():
    with pytest.raises(FileNotFoundError):
        profile_pdf("non_existent_file_12345.pdf")

