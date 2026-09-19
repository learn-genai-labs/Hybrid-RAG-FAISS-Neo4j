import json

from langchain_openai import ChatOpenAI

from backend.config.settings import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)
from backend.evaluation.evaluation_models import (
    EvaluationMetric,
    RAGEvaluation,
)
from backend.retrieval.hybrid_retriever import (
    format_hybrid_context,
)


def _clamp_score(value) -> float:
    """
    Ensure evaluation score stays between 0 and 1.
    """

    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(
        0.0,
        min(1.0, score),
    )


def evaluate_rag_response(
    question: str,
    answer: str,
    retrieval_result: dict,
) -> dict:
    """
    Evaluate a Hybrid RAG response using an LLM judge.

    Metrics:
        1. Retrieval relevance
        2. Faithfulness
        3. Answer relevance
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    hybrid_context = format_hybrid_context(
        retrieval_result
    )

    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    prompt = f"""
You are evaluating a Hybrid RAG system.

Evaluate the answer strictly against the supplied question
and retrieved context.

Do not use external knowledge.

Return valid JSON only.

Use exactly this structure:

{{
  "retrieval_relevance": {{
    "score": 0.0,
    "reason": "short explanation"
  }},
  "faithfulness": {{
    "score": 0.0,
    "reason": "short explanation"
  }},
  "answer_relevance": {{
    "score": 0.0,
    "reason": "short explanation"
  }}
}}

Scoring:

0.0 = poor
0.5 = partially satisfactory
1.0 = strong

Scores may be decimals between 0 and 1.

Definitions:

RETRIEVAL RELEVANCE:
How relevant is the retrieved context to the user's question?

FAITHFULNESS:
Is the answer supported by the retrieved context?
Penalize unsupported claims.

ANSWER RELEVANCE:
Does the answer directly and sufficiently address the
user's question?

QUESTION:

{question}

RETRIEVED CONTEXT:

{hybrid_context}

ANSWER:

{answer}
""".strip()

    response = llm.invoke(
        prompt
    )

    response_text = response.content

    if not isinstance(
        response_text,
        str,
    ):
        raise ValueError(
            "Evaluator returned an unexpected response."
        )

    cleaned = response_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]

    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    try:
        evaluation_data = json.loads(
            cleaned
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "Evaluator did not return valid JSON.\n"
            f"Response:\n{response_text}"
        ) from error

    retrieval = evaluation_data.get(
        "retrieval_relevance",
        {},
    )

    faithfulness = evaluation_data.get(
        "faithfulness",
        {},
    )

    relevance = evaluation_data.get(
        "answer_relevance",
        {},
    )

    evaluation = RAGEvaluation(
        retrieval_relevance=EvaluationMetric(
            name="Retrieval Relevance",
            score=_clamp_score(
                retrieval.get("score")
            ),
            reason=retrieval.get(
                "reason",
                "",
            ),
        ),
        faithfulness=EvaluationMetric(
            name="Faithfulness",
            score=_clamp_score(
                faithfulness.get("score")
            ),
            reason=faithfulness.get(
                "reason",
                "",
            ),
        ),
        answer_relevance=EvaluationMetric(
            name="Answer Relevance",
            score=_clamp_score(
                relevance.get("score")
            ),
            reason=relevance.get(
                "reason",
                "",
            ),
        ),
    )

    result = evaluation.to_dict()

    result["average_score"] = round(
        (
            evaluation.retrieval_relevance.score
            + evaluation.faithfulness.score
            + evaluation.answer_relevance.score
        )
        / 3,
        3,
    )

    return result