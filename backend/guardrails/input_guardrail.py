import re


MAX_QUESTION_LENGTH = 1000

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(the\s+)?system\s+prompt",
    r"print\s+(the\s+)?system\s+prompt",
    r"developer\s+message",
    r"bypass\s+(the\s+)?instructions",
    r"override\s+(the\s+)?instructions",
]


def validate_question(question: str) -> dict:
    """
    Apply deterministic input guardrails.

    Returns:
        {
            "allowed": bool,
            "reason": str | None
        }
    """

    if not question:
        return {
            "allowed": False,
            "reason": "Question cannot be empty.",
        }

    question = question.strip()

    if not question:
        return {
            "allowed": False,
            "reason": "Question cannot be empty.",
        }

    if len(question) > MAX_QUESTION_LENGTH:
        return {
            "allowed": False,
            "reason": (
                f"Question exceeds the maximum "
                f"length of {MAX_QUESTION_LENGTH} characters."
            ),
        }

    lowered_question = question.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:

        if re.search(
            pattern,
            lowered_question,
            flags=re.IGNORECASE,
        ):
            return {
                "allowed": False,
                "reason": (
                    "The question contains instructions "
                    "that attempt to override the RAG system."
                ),
            }

    return {
        "allowed": True,
        "reason": None,
    }