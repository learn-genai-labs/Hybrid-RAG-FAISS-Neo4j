FALLBACK_ANSWER = (
    "I could not generate a sufficiently grounded answer "
    "from the ingested document."
)


def validate_answer(
    answer: str,
    retrieval_result: dict,
) -> dict:
    """
    Apply basic deterministic output checks.

    Detailed semantic groundedness is handled separately
    by the evaluation layer.
    """

    if not answer:
        return {
            "allowed": False,
            "reason": "The model returned an empty answer.",
            "answer": FALLBACK_ANSWER,
        }

    answer = answer.strip()

    if not answer:
        return {
            "allowed": False,
            "reason": "The model returned an empty answer.",
            "answer": FALLBACK_ANSWER,
        }

    vector_results = retrieval_result.get(
        "vector_results",
        [],
    )

    graph_results = retrieval_result.get(
        "graph_results",
        [],
    )

    if not vector_results and not graph_results:

        return {
            "allowed": False,
            "reason": (
                "The answer has no retrieved evidence."
            ),
            "answer": FALLBACK_ANSWER,
        }

    return {
        "allowed": True,
        "reason": None,
        "answer": answer,
    }