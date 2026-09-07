from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=1200, chunk_overlap=180):
    """
    Split text into larger, more coherent chunks while preserving
    paragraphs and sentence boundaries as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". "
        ],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return cleaned_chunks