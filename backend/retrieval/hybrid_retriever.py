from backend.retrieval.vector_retriever import (
    search_faiss,
)
from backend.retrieval.graph_retriever import (
    search_knowledge_graph,
)


def retrieve_hybrid_context(
    query: str,
    vector_top_k: int = 4,
    graph_limit: int = 10,
) -> dict:
    """
    Retrieve context from both FAISS and Neo4j.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    # ---------------------------------------------------------
    # Vector retrieval
    # ---------------------------------------------------------

    vector_results = search_faiss(
        query=query,
        top_k=vector_top_k,
    )

    vector_context = []

    for document, score in vector_results:

        vector_context.append(
            {
                "content": document.page_content,
                "metadata": document.metadata,
                "score": float(score),
            }
        )

    # ---------------------------------------------------------
    # Knowledge Graph retrieval
    # ---------------------------------------------------------

    graph_results = search_knowledge_graph(
        user_question=query,
        limit=graph_limit,
    )

    return {
        "query": query,
        "vector_results": vector_context,
        "graph_results": graph_results,
    }


def format_hybrid_context(
    retrieval_result: dict,
) -> str:
    """
    Convert vector and graph retrieval results into
    context suitable for the LLM.
    """

    sections = []

    # ---------------------------------------------------------
    # Vector context
    # ---------------------------------------------------------

    sections.append(
        "VECTOR RETRIEVAL CONTEXT"
    )

    sections.append(
        "=" * 50
    )

    for index, result in enumerate(
        retrieval_result["vector_results"],
        start=1,
    ):

        metadata = result["metadata"]

        sections.append(
            f"""
Vector Result {index}
Source: {metadata.get('file_name')}
Page: {metadata.get('page')}
Chunk: {metadata.get('chunk_id')}

{result['content']}
""".strip()
        )

    # ---------------------------------------------------------
    # Graph context
    # ---------------------------------------------------------

    sections.append(
        "\nKNOWLEDGE GRAPH CONTEXT"
    )

    sections.append(
        "=" * 50
    )

    graph_results = retrieval_result[
        "graph_results"
    ]

    if graph_results:

        for result in graph_results:

            sections.append(
                (
                    f"{result['source']} "
                    f"--[{result['relationship']}]--> "
                    f"{result['target']}"
                )
            )

    else:

        sections.append(
            "No directly matching graph relationships found."
        )

    return "\n\n".join(sections)


if __name__ == "__main__":

    question = (
        "What databases and storage technologies "
        "are used in the architecture?"
    )

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    result = retrieve_hybrid_context(
        question
    )

    print(
        f"\nVector results: "
        f"{len(result['vector_results'])}"
    )

    print(
        f"Graph results: "
        f"{len(result['graph_results'])}"
    )

    print("\n")

    print(
        format_hybrid_context(result)
    )