from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    force: bool = False


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["skill", "hybrid", "vector"] = "hybrid"
    top_k: int = Field(default=8, ge=1, le=25)
    session_id: str | None = Field(default=None, min_length=1)
    actor_id: str | None = Field(default=None, min_length=1)


class EvaluateRequest(BaseModel):
    mode: Literal["skill", "hybrid", "vector"] = "hybrid"


class SkillExecuteRequest(BaseModel):
    skill_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=25)


class CustomerServiceGapResolveRequest(BaseModel):
    answer: str = Field(min_length=1)
    reviewer: str | None = Field(default=None, min_length=1)
    label: str | None = Field(default="ai_customer_service", min_length=1)
    question: str | None = Field(default=None, min_length=1)
    url: str | None = Field(default=None, min_length=1)
    auto_ingest: bool = True


class GenericResponse(BaseModel):
    payload: dict[str, Any]
