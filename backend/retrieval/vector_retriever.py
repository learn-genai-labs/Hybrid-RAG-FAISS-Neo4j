from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from backend.config.settings import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
)


def load_faiss_vector_store() -> FAISS:
    """
    Load the persisted FAISS vector store from disk.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured in the .env file."
        )

    if not FAISS_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found: {FAISS_INDEX_DIR}"
        )

    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    vector_store = FAISS.load_local(
        folder_path=str(FAISS_INDEX_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def search_faiss(
    query: str,
    top_k: int = 3,
):
    """
    Search FAISS and return the most relevant document chunks.
    """

    if not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    vector_store = load_faiss_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    return results


if __name__ == "__main__":

    test_question = (
        "What databases and storage technologies "
        "are used in the architecture?"
    )

    print("\n" + "=" * 70)
    print("FAISS VECTOR RETRIEVAL TEST")
    print("=" * 70)

    print(f"\nQuestion:\n{test_question}")

    results = search_faiss(
        query=test_question,
        top_k=3,
    )

    print(
        f"\nRetrieved chunks: {len(results)}"
    )

    for rank, (document, score) in enumerate(
        results,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(f"RESULT {rank}")
        print("=" * 70)

        print(f"\nFAISS distance score: {score}")

        print("\nMetadata:")
        print(document.metadata)

        print("\nContent:")
        print("-" * 70)
        print(document.page_content)
        print("-" * 70)