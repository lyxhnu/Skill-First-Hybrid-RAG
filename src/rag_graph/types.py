from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    evidence_id: str
    source_path: str
    file_type: str
    location: dict[str, Any]
    content: str
    retrieval_source: str
    score: float
    domain: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "location": self.location,
            "content": self.content,
            "retrieval_source": self.retrieval_source,
            "score": self.score,
            "domain": self.domain,
            "metadata": self.metadata,
        }


def evidence_from_dict(payload: dict[str, Any]) -> Evidence:
    return Evidence(
        evidence_id=payload["evidence_id"],
        source_path=payload["source_path"],
        file_type=payload["file_type"],
        location=payload.get("location", {}),
        content=payload["content"],
        retrieval_source=payload.get("retrieval_source", "unknown"),
        score=float(payload.get("score", 0.0)),
        domain=payload.get("domain", ""),
        metadata=payload.get("metadata", {}),
    )


@dataclass
class QueryConstraintPlan:
    raw_query: str
    hard_terms: list[str] = field(default_factory=list)
    soft_terms: list[str] = field(default_factory=list)
    intent: str = "lookup"
    answer_shape: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "hard_terms": list(self.hard_terms),
            "soft_terms": list(self.soft_terms),
            "intent": self.intent,
            "answer_shape": self.answer_shape,
            "metadata": dict(self.metadata),
        }


def query_plan_from_dict(payload: dict[str, Any]) -> QueryConstraintPlan:
    return QueryConstraintPlan(
        raw_query=str(payload.get("raw_query", "")),
        hard_terms=[str(item) for item in payload.get("hard_terms", []) if str(item).strip()],
        soft_terms=[str(item) for item in payload.get("soft_terms", []) if str(item).strip()],
        intent=str(payload.get("intent", "lookup") or "lookup"),
        answer_shape=str(payload.get("answer_shape", "unknown") or "unknown"),
        metadata=dict(payload.get("metadata", {})),
    )
