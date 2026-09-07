import hashlib
from pathlib import Path


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate a stable SHA-256 hash for the source file.
    """
    path = Path(file_path)

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def make_document_id(file_hash: str) -> str:
    """
    Create a stable document identifier from the file hash.
    """
    return f"doc_{file_hash[:16]}"


def make_element_id(
    document_id: str,
    page_number: int,
    element_index: int,
) -> str:
    """
    Create a deterministic identifier for a page element.
    """
    return f"{document_id}_p{page_number}_e{element_index}"
