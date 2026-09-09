import contextlib
import os
import re
import time
from typing import Any
import pymupdf

from src.document.models import DocumentProfile

# Common English stopwords for lightweight lexical language detection
ENGLISH_STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have",
    "it", "for", "not", "on", "with", "as", "at", "this", "by",
    "from", "they", "an", "policy", "is", "are", "which", "or",
    "scheme", "semiconductor", "government", "india", "manufacturing"
}

MIN_CHAR_THRESHOLD = 50


def detect_language_hint(sample_text: str) -> str:
    """
    Fast, deterministic language signal without heavy NLP dependencies.
    """
    if not sample_text:
        return "unknown"

    words = re.findall(r"\b[a-zA-Z]{2,}\b", sample_text.lower())
    if not words:
        return "unknown"

    matches = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    if len(words) >= 8 and (matches / len(words)) >= 0.05:
        return "en"

    return "unknown"


def profile_pdf(file_path: str, min_char_threshold: int = MIN_CHAR_THRESHOLD) -> DocumentProfile:
    """
    Quickly inspect a PDF using PyMuPDF to extract structural and quality signals.

    Produces a DocumentProfile without running heavy OCR, LLM, or vision models.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    start_time = time.perf_counter()

    pdf = pymupdf.open(file_path)

    try:
        page_count = len(pdf)
        if page_count == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return DocumentProfile(
                file_path=os.path.abspath(file_path),
                page_count=0,
                text_coverage=0.0,
                avg_chars_per_page=0.0,
                scanned_page_ratio=0.0,
                total_characters=0,
                total_images=0,
                images_per_page=0.0,
                total_text_blocks=0,
                avg_blocks_per_page=0.0,
                estimated_table_count=0,
                has_complex_layout=False,
                detected_language="unknown",
                profiling_time_ms=elapsed_ms,
                metadata={"empty_document": True}
            )

        pages_with_meaningful_text = 0
        scanned_pages = 0
        total_characters = 0
        total_images = 0
        total_text_blocks = 0
        total_tables = 0
        sample_texts = []

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                for page in pdf:
                    # 1. Extract text and character count
                    text = page.get_text("text").strip()
                    char_count = len(text)
                    total_characters += char_count

                    if char_count >= min_char_threshold:
                        pages_with_meaningful_text += 1
                    else:
                        scanned_pages += 1

                    if len(sample_texts) < 5 and char_count > 0:
                        sample_texts.append(text[:500])

                    # 2. Extract image count
                    try:
                        images = page.get_images(full=True)
                        total_images += len(images)
                    except Exception:
                        pass

                    # 3. Extract text blocks
                    try:
                        blocks = page.get_text("blocks")
                        text_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0]
                        total_text_blocks += len(text_blocks) if text_blocks else len(blocks)
                    except Exception:
                        pass

                    # 4. Extract table count (native PyMuPDF find_tables)
                    if hasattr(page, "find_tables"):
                        try:
                            tables = page.find_tables()
                            total_tables += len(tables.tables)
                        except Exception:
                            pass

        text_coverage = pages_with_meaningful_text / page_count
        scanned_page_ratio = scanned_pages / page_count
        avg_chars_per_page = total_characters / page_count
        images_per_page = total_images / page_count
        avg_blocks_per_page = total_text_blocks / page_count

        has_complex_layout = (total_tables > 0) or (avg_blocks_per_page > 15.0)

        combined_sample = " ".join(sample_texts)
        detected_language = detect_language_hint(combined_sample)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return DocumentProfile(
            file_path=os.path.abspath(file_path),
            page_count=page_count,
            text_coverage=text_coverage,
            avg_chars_per_page=avg_chars_per_page,
            scanned_page_ratio=scanned_page_ratio,
            total_characters=total_characters,
            total_images=total_images,
            images_per_page=images_per_page,
            total_text_blocks=total_text_blocks,
            avg_blocks_per_page=avg_blocks_per_page,
            estimated_table_count=total_tables,
            has_complex_layout=has_complex_layout,
            detected_language=detected_language,
            profiling_time_ms=elapsed_ms,
            metadata={
                "meaningful_text_pages": pages_with_meaningful_text,
                "scanned_pages": scanned_pages,
            }
        )

    finally:
        pdf.close()
