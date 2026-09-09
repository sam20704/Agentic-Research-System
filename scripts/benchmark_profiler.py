import sys
sys.path.insert(0, ".")
import os
from pathlib import Path
from src.ingestion.profile import profile_pdf


def run_benchmark():
    ref_dir = Path("data/references")
    pdf_files = sorted(ref_dir.glob("*.pdf"))

    if not pdf_files:
        print("No reference PDFs found in data/references.")
        return

    print("=" * 105)
    print(f"{'PDF File':<45} | {'Pages':>5} | {'Time (ms)':>9} | {'ms/page':>7} | {'Txt Cov':>7} | {'Avg Chars':>9} | {'Scan %':>6} | {'Img/pg':>6}")
    print("=" * 105)

    total_pages = 0
    total_time_ms = 0.0

    records = []

    for pdf_path in pdf_files:
        profile = profile_pdf(str(pdf_path))
        records.append(profile)

        ms_per_page = profile.profiling_time_ms / profile.page_count if profile.page_count else 0
        total_pages += profile.page_count
        total_time_ms += profile.profiling_time_ms

        name = pdf_path.name[:43] + ".." if len(pdf_path.name) > 45 else pdf_path.name
        print(
            f"{name:<45} | "
            f"{profile.page_count:>5} | "
            f"{profile.profiling_time_ms:>9.1f} | "
            f"{ms_per_page:>7.2f} | "
            f"{profile.text_coverage * 100:>6.1f}% | "
            f"{profile.avg_chars_per_page:>9.0f} | "
            f"{profile.scanned_page_ratio * 100:>5.1f}% | "
            f"{profile.images_per_page:>6.2f}"
        )

    print("=" * 105)
    avg_speed = total_time_ms / total_pages if total_pages else 0
    pages_per_sec = (total_pages / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0
    print(f"Total PDFs profiled: {len(pdf_files)}")
    print(f"Total Pages:         {total_pages}")
    print(f"Total Time:          {total_time_ms:.1f} ms ({total_time_ms / 1000.0:.2f} s)")
    print(f"Average per page:    {avg_speed:.2f} ms/page ({pages_per_sec:.1f} pages/sec)")
    print("=" * 105)


if __name__ == "__main__":
    run_benchmark()
