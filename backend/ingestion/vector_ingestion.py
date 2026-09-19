from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from backend.config.settings import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
)
from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.text_chunker import chunk_documents


def create_faiss_vector_store(
    chunks: list[Document],
) -> FAISS:
    """
    Create embeddings for document chunks and store them
    in a FAISS vector store.
    """

    if not chunks:
        raise ValueError(
            "No chunks were provided for FAISS ingestion."
        )

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured in the .env file."
        )

    print(
        f"\nCreating embeddings using: "
        f"{OPENAI_EMBEDDING_MODEL}"
    )

    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    print(
        f"Creating embeddings for {len(chunks)} chunks..."
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def save_faiss_vector_store(
    vector_store: FAISS,
    index_directory: Path = FAISS_INDEX_DIR,
) -> None:
    """
    Save the FAISS vector store locally.
    """

    index_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(
        str(index_directory)
    )

    print(
        f"\nFAISS vector store saved successfully."
    )

    print(
        f"Location: {index_directory}"
    )


def build_faiss_index(
    pdf_path: str,
) -> FAISS:
    """
    Complete vector ingestion flow:

    PDF
      -> Load pages
      -> Chunk documents
      -> Generate embeddings
      -> Create FAISS index
      -> Save FAISS index
    """

    print("\n" + "=" * 70)
    print("FAISS VECTOR INGESTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1 - Load PDF
    # ---------------------------------------------------------

    print("\n1. Loading PDF...")

    documents = load_pdf(pdf_path)

    print(
        f"   Pages loaded: {len(documents)}"
    )

    # ---------------------------------------------------------
    # Step 2 - Chunk documents
    # ---------------------------------------------------------

    print("\n2. Chunking documents...")

    chunks = chunk_documents(documents)

    print(
        f"   Chunks created: {len(chunks)}"
    )

    # ---------------------------------------------------------
    # Step 3 - Create FAISS vector store
    # ---------------------------------------------------------

    print("\n3. Creating vector embeddings...")

    vector_store = create_faiss_vector_store(
        chunks
    )

    # ---------------------------------------------------------
    # Step 4 - Save FAISS index
    # ---------------------------------------------------------

    print("\n4. Saving FAISS index...")

    save_faiss_vector_store(
        vector_store
    )

    print("\n" + "=" * 70)
    print("VECTOR INGESTION COMPLETED")
    print("=" * 70)

    return vector_store


# -------------------------------------------------------------
# Local test
# -------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            "python -m backend.ingestion.vector_ingestion "
            "\"data\\uploads\\your_file.pdf\"\n"
        )

        raise SystemExit(1)

    pdf_file = sys.argv[1]

    try:

        build_faiss_index(
            pdf_file
        )

    except Exception as error:

        print("\nVector ingestion failed.")
        print(f"Error: {error}")

        raise