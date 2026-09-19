def validate_retrieval_context(
    retrieval_result: dict,
) -> dict:
    """
    Check whether Hybrid RAG retrieved enough evidence
    to proceed with answer generation.
    """

    vector_results = retrieval_result.get(
        "vector_results",
        [],
    )

    graph_results = retrieval_result.get(
        "graph_results",
        [],
    )

    has_vector_context = (
        len(vector_results) > 0
    )

    has_graph_context = (
        len(graph_results) > 0
    )

    if not has_vector_context and not has_graph_context:

        return {
            "allowed": False,
            "reason": (
                "No relevant information was retrieved "
                "from FAISS or Neo4j."
            ),
        }

    return {
        "allowed": True,
        "reason": None,
        "vector_context_available": (
            has_vector_context
        ),
        "graph_context_available": (
            has_graph_context
        ),
    }