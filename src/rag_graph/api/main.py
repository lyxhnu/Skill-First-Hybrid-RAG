from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from ..service import RAGService
from .schemas import (
    CustomerServiceGapResolveRequest,
    EvaluateRequest,
    GenericResponse,
    IngestRequest,
    QueryRequest,
    SkillExecuteRequest,
)


app = FastAPI(title="Skill-First Hybrid RAG", version="0.1.0")
service = RAGService()
STATIC_DIR = Path(__file__).resolve().parent / "static"
CHAT_PAGE = STATIC_DIR / "index.html"


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(CHAT_PAGE)


@app.get("/chat", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(CHAT_PAGE)


@app.get("/health", response_model=GenericResponse)
def health() -> GenericResponse:
    return GenericResponse(payload=service.health())


@app.post("/ingest", response_model=GenericResponse)
def ingest(request: IngestRequest) -> GenericResponse:
    payload = service.ingest(force=request.force)
    return GenericResponse(payload=payload)


@app.post("/query", response_model=GenericResponse)
def query(request: QueryRequest) -> GenericResponse:
    try:
        payload = service.query(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            session_id=request.session_id,
            actor_id=request.actor_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GenericResponse(payload=payload)


@app.post("/evaluate", response_model=GenericResponse)
def evaluate(request: EvaluateRequest) -> GenericResponse:
    payload = service.evaluate(mode=request.mode)
    return GenericResponse(payload=payload)


@app.get("/skills", response_model=GenericResponse)
def list_skills() -> GenericResponse:
    payload = {"skills": service.list_skills()}
    return GenericResponse(payload=payload)


@app.post("/skills/execute", response_model=GenericResponse)
def execute_skill(request: SkillExecuteRequest) -> GenericResponse:
    try:
        payload = service.execute_skill(skill_id=request.skill_id, query=request.query, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GenericResponse(payload=payload)


@app.get("/customer-service/gaps", response_model=GenericResponse)
def list_customer_service_gaps(
    status: str = Query(default="open"),
    limit: int = Query(default=100, ge=1, le=500),
) -> GenericResponse:
    try:
        payload = service.list_customer_service_gaps(status=status, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GenericResponse(payload=payload)


@app.post("/customer-service/gaps/{gap_id}/resolve", response_model=GenericResponse)
def resolve_customer_service_gap(
    gap_id: str,
    request: CustomerServiceGapResolveRequest,
) -> GenericResponse:
    try:
        payload = service.resolve_customer_service_gap(
            gap_id=gap_id,
            answer=request.answer,
            reviewer=request.reviewer,
            label=request.label,
            question=request.question,
            url=request.url,
            auto_ingest=request.auto_ingest,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GenericResponse(payload=payload)
