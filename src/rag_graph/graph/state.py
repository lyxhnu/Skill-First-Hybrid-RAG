from __future__ import annotations

from typing import Any

from typing_extensions import NotRequired, TypedDict


class RAGState(TypedDict):
    query: str
    mode: str
    top_k: int
    session_id: NotRequired[str]
    actor_id: NotRequired[str]
    effective_query: NotRequired[str]
    memory_context: NotRequired[dict[str, Any]]
    memory_trace: NotRequired[dict[str, Any]]
    query_intent: NotRequired[dict[str, Any]]
    query_constraints: NotRequired[dict[str, Any]]
    selected_skills: NotRequired[list[str]]
    candidate_dirs: NotRequired[list[str]]
    candidate_files: NotRequired[list[str]]
    skill_evidence: NotRequired[list[dict[str, Any]]]
    vector_evidence: NotRequired[list[dict[str, Any]]]
    merged_evidence: NotRequired[list[dict[str, Any]]]
    reranked_evidence: NotRequired[list[dict[str, Any]]]
    iteration_count: NotRequired[int]
    confidence: NotRequired[float]
    need_vector: NotRequired[bool]
    selected_models: NotRequired[dict[str, str]]
    answer: NotRequired[str]
    answerable: NotRequired[bool]
    citations: NotRequired[list[dict[str, Any]]]
    answer_support: NotRequired[dict[str, Any]]
    errors: NotRequired[list[str]]
