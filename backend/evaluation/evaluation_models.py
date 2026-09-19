from dataclasses import dataclass, asdict


@dataclass
class EvaluationMetric:
    name: str
    score: float
    reason: str


@dataclass
class RAGEvaluation:
    retrieval_relevance: EvaluationMetric
    faithfulness: EvaluationMetric
    answer_relevance: EvaluationMetric

    def to_dict(self) -> dict:
        return asdict(self)