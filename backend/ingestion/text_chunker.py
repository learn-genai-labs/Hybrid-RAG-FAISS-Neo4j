from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.ingestion.pdf_loader import load_pdf


# -------------------------------------------------------------
# Chunking configuration
# -------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split LangChain Documents into smaller chunks.

    Metadata from the original PDF pages is preserved in
    every generated chunk.

    Parameters:
        documents:
            Documents produced by pdf_loader.py.

        chunk_size:
            Maximum approximate character size of each chunk.

        chunk_overlap:
            Number of overlapping characters between
            consecutive chunks.

    Returns:
        List of chunked LangChain Documents.
    """

    if not documents:
        raise ValueError(
            "No documents were provided for chunking."
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    # ---------------------------------------------------------
    # Create text splitter
    # ---------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    # ---------------------------------------------------------
    # Split documents
    # ---------------------------------------------------------

    chunks = text_splitter.split_documents(documents)

    # ---------------------------------------------------------
    # Add chunk metadata
    # ---------------------------------------------------------

    for chunk_id, chunk in enumerate(chunks, start=1):

        chunk.metadata["chunk_id"] = chunk_id

    return chunks


# -------------------------------------------------------------
# Local test
# -------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            "python -m backend.ingestion.text_chunker "
            "\"data\\uploads\\your_file.pdf\"\n"
        )

        raise SystemExit(1)

    pdf_file = sys.argv[1]

    print("\nLoading PDF...")

    documents = load_pdf(pdf_file)

    print(
        f"Pages loaded: {len(documents)}"
    )

    print("\nChunking documents...")

    chunks = chunk_documents(documents)

    print("\nChunking completed successfully.")

    print(
        f"Total chunks created: {len(chunks)}"
    )

    print(
        f"Chunk size: {DEFAULT_CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {DEFAULT_CHUNK_OVERLAP}"
    )

    # ---------------------------------------------------------
    # Display sample chunks
    # ---------------------------------------------------------

    sample_count = min(3, len(chunks))

    for index in range(sample_count):

        chunk = chunks[index]

        print("\n" + "=" * 70)

        print(
            f"CHUNK {index + 1}"
        )

        print("=" * 70)

        print("\nMetadata:")
        print(chunk.metadata)

        print("\nContent:")
        print("-" * 70)

        print(chunk.page_content)

        print("-" * 70)

        print(
            f"Characters: {len(chunk.page_content)}"
        )