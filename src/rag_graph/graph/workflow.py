from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..fusion.fuse import EvidenceFusion
from ..memory.manager import MemoryManager
from ..models.providers import ModelGateway, RerankGateway
from ..query_runtime.analyzer import QueryConstraintAnalyzer
from ..skill_runtime.manager import SkillManager
from ..skill_runtime.router import SkillRouter
from ..types import QueryConstraintPlan, query_plan_from_dict
from ..utils.text import extract_keywords, lexical_score, normalize_text
from ..vector_store.index import VectorStore
from .state import RAGState


def build_workflow(
    *,
    settings: Settings,
    router: SkillRouter,
    skill_manager: SkillManager,
    vector_store: VectorStore,
    fusion: EvidenceFusion,
    model_gateway: ModelGateway,
    rerank_gateway: RerankGateway,
    query_analyzer: QueryConstraintAnalyzer,
    memory_manager: MemoryManager,
) -> Callable[[RAGState], RAGState]:
    graph = StateGraph(RAGState)

    def load_memory_context(state: RAGState) -> dict[str, Any]:
        session_id = str(state.get("session_id") or "default-session")
        actor_id = str(state.get("actor_id") or "default-user")
        context = memory_manager.build_context(session_id=session_id, actor_id=actor_id, query=state["query"])
        return {
            "session_id": session_id,
            "actor_id": actor_id,
            "effective_query": context["effective_query"],
            "memory_context": {"prompt": context["prompt"]},
            "memory_trace": context["trace"],
        }

    def analyze_query(state: RAGState) -> dict[str, Any]:
        active_query = str(state.get("effective_query") or state["query"])
        keywords = extract_keywords(active_query)
        query_plan = query_analyzer.analyze(active_query)
        selected_skills = skill_manager.select_skills(active_query, top_n=3)
        return {
            "query_intent": {
                "keywords": keywords,
                "intent": query_plan.intent,
                "answer_shape": query_plan.answer_shape,
            },
            "query_constraints": query_plan.to_dict(),
            "selected_skills": selected_skills,
            "selected_models": {
                "chat_provider": settings.chat_provider,
                "chat_model": settings.chat_model,
                "embed_provider": settings.embed_provider,
                "embed_model": settings.embed_model,
                "rerank_provider": settings.rerank_provider,
                "rerank_model": settings.rerank_model,
            },
            "errors": [],
        }

    def route_by_skill_index(state: RAGState) -> dict[str, Any]:
        active_query = str(state.get("effective_query") or state["query"])
        candidate_dirs, candidate_files = router.route(
            active_query,
            query_plan=state.get("query_constraints"),
            max_domains=2,
            max_files=12,
        )
        return {"candidate_dirs": candidate_dirs, "candidate_files": candidate_files}

    def refine_query_plan(state: RAGState) -> dict[str, Any]:
        active_query = str(state.get("effective_query") or state["query"])
        refined = query_analyzer.refine_for_files(
            active_query,
            state.get("query_constraints"),
            state.get("candidate_files", []),
        )
        return {"query_constraints": refined.to_dict()}

    def run_skill_retrieval(state: RAGState) -> dict[str, Any]:
        mode = state["mode"]
        if mode == "vector":
            return {"skill_evidence": [], "iteration_count": 0}
        active_query = str(state.get("effective_query") or state["query"])
        evidence = skill_manager.retrieve_for_query(
            query=active_query,
            query_plan=state.get("query_constraints"),
            selected_skills=state.get("selected_skills", []),
            candidate_files=state.get("candidate_files", []),
            top_k=state["top_k"],
        )
        return {"skill_evidence": evidence, "iteration_count": 1}

    def assess_evidence(state: RAGState) -> dict[str, Any]:
        mode = state["mode"]
        skill_evidence = state.get("skill_evidence", [])
        if mode == "vector":
            return {"confidence": 0.0, "need_vector": True}
        if not skill_evidence:
            return {"confidence": 0.0, "need_vector": mode == "hybrid"}
        top_score = float(skill_evidence[0]["score"])
        enough_hits = len(skill_evidence) >= settings.skill_min_hits
        enough_score = top_score >= settings.skill_confidence_threshold
        confidence = top_score if enough_score else top_score * 0.6
        need_vector = mode == "hybrid" and not (enough_hits and enough_score)
        return {"confidence": confidence, "need_vector": need_vector}

    def run_vector_retrieval(state: RAGState) -> dict[str, Any]:
        mode = state["mode"]
        allowed_files = None
        if mode in {"hybrid", "vector"}:
            files = state.get("candidate_files", [])
            if files:
                allowed_files = set(files)
        active_query = str(state.get("effective_query") or state["query"])
        evidence = vector_store.search(query=active_query, top_k=state["top_k"], allowed_files=allowed_files)
        return {"vector_evidence": evidence}

    def fuse_evidence(state: RAGState) -> dict[str, Any]:
        merged = fusion.fuse(
            skill_evidence=state.get("skill_evidence", []),
            vector_evidence=state.get("vector_evidence", []),
            mode=state["mode"],
            top_k=state["top_k"],
        )
        return {"merged_evidence": merged}

    def rerank_evidence(state: RAGState) -> dict[str, Any]:
        active_query = str(state.get("effective_query") or state["query"])
        reranked = rerank_gateway.rerank(
            query=active_query,
            evidence=state.get("merged_evidence", []),
            top_k=state["top_k"],
            query_plan=state.get("query_constraints"),
        )
        return {"reranked_evidence": reranked}

    def generate_answer(state: RAGState) -> dict[str, Any]:
        evidence = _generation_evidence(
            state.get("reranked_evidence", []),
            state.get("merged_evidence", []),
            limit=max(state["top_k"], 10),
        )
        if not evidence:
            return {
                "answer": "未在知识库中检索到相关内容，无法回答该问题。请补充更具体的关键词、文件名或时间范围。",
                "citations": [],
                "answerable": False,
                "answer_support": {
                    "explicit": False,
                    "reason": "no_evidence",
                    "max_support_score": 0.0,
                    "threshold": _explicit_support_threshold(None),
                },
            }
        direct_answer = _exact_qa_record_answer(state["query"], evidence)
        if direct_answer is not None:
            return direct_answer
        support = _explicit_answer_support(
            query=state["query"],
            evidence=evidence,
            query_plan=state.get("query_constraints"),
        )
        if not bool(support.get("explicit", False)):
            return {
                "answer": "检索到的内容不足以明确支持该问题，无法可靠回答。请补充更具体的实体、文件名、时间范围或业务条件。",
                "citations": [],
                "answerable": False,
                "answer_support": support,
            }
        answer = model_gateway.generate(
            state["query"],
            evidence,
            query_plan=state.get("query_constraints"),
            memory_context=str(state.get("memory_context", {}).get("prompt", "")),
        )
        citations: list[dict[str, Any]] = []
        for item in evidence[:5]:
            citations.append(
                {
                    "evidence_id": item["evidence_id"],
                    "source_path": item["source_path"],
                    "location": item["location"],
                    "retrieval_source": item["retrieval_source"],
                }
            )
        return {"answer": answer, "citations": citations, "answerable": True, "answer_support": support}

    def verify_citations(state: RAGState) -> dict[str, Any]:
        if state.get("answerable") is False:
            return {"citations": []}
        evidence_pool = state.get("reranked_evidence", state.get("merged_evidence", []))
        valid_ids = {item["evidence_id"] for item in evidence_pool}
        citations = [row for row in state.get("citations", []) if row["evidence_id"] in valid_ids]
        if not citations and evidence_pool:
            item = evidence_pool[0]
            citations = [
                {
                    "evidence_id": item["evidence_id"],
                    "source_path": item["source_path"],
                    "location": item["location"],
                    "retrieval_source": item["retrieval_source"],
                }
            ]
        return {"citations": citations}

    def persist_memory(state: RAGState) -> dict[str, Any]:
        summary_ids = memory_manager.persist_turn(
            session_id=str(state.get("session_id") or "default-session"),
            actor_id=str(state.get("actor_id") or "default-user"),
            user_query=state["query"],
            effective_query=str(state.get("effective_query") or state["query"]),
            answer=str(state.get("answer", "")),
            citations=list(state.get("citations", [])),
            query_constraints=dict(state.get("query_constraints", {})),
        )
        memory_trace = dict(state.get("memory_trace", {}))
        memory_trace.update(summary_ids)
        return {"memory_trace": memory_trace}

    def finalize_response(state: RAGState) -> dict[str, Any]:
        return {}

    def choose_after_assess(state: RAGState) -> str:
        mode = state["mode"]
        if mode == "skill":
            return "fuse_evidence"
        if mode == "vector":
            return "run_vector_retrieval"
        return "run_vector_retrieval" if state.get("need_vector", False) else "fuse_evidence"

    graph.add_node("load_memory_context", load_memory_context)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("route_by_skill_index", route_by_skill_index)
    graph.add_node("refine_query_plan", refine_query_plan)
    graph.add_node("run_skill_retrieval", run_skill_retrieval)
    graph.add_node("assess_evidence", assess_evidence)
    graph.add_node("run_vector_retrieval", run_vector_retrieval)
    graph.add_node("fuse_evidence", fuse_evidence)
    graph.add_node("rerank_evidence", rerank_evidence)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_citations", verify_citations)
    graph.add_node("persist_memory", persist_memory)
    graph.add_node("finalize_response", finalize_response)

    graph.add_edge(START, "load_memory_context")
    graph.add_edge("load_memory_context", "analyze_query")
    graph.add_edge("analyze_query", "route_by_skill_index")
    graph.add_edge("route_by_skill_index", "refine_query_plan")
    graph.add_edge("refine_query_plan", "run_skill_retrieval")
    graph.add_edge("run_skill_retrieval", "assess_evidence")
    graph.add_conditional_edges(
        "assess_evidence",
        choose_after_assess,
        {
            "run_vector_retrieval": "run_vector_retrieval",
            "fuse_evidence": "fuse_evidence",
        },
    )
    graph.add_edge("run_vector_retrieval", "fuse_evidence")
    graph.add_edge("fuse_evidence", "rerank_evidence")
    graph.add_edge("rerank_evidence", "generate_answer")
    graph.add_edge("generate_answer", "verify_citations")
    graph.add_edge("verify_citations", "persist_memory")
    graph.add_edge("persist_memory", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph.compile()



def _exact_qa_record_answer(query: str, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized_query = _normalize_qa_text(query)
    if not normalized_query:
        return None

    best_item: dict[str, Any] | None = None
    best_answer = ""
    best_score = 0.0

    for item in evidence:
        if str(item.get("file_type", "")).lower() != "json":
            continue
        metadata = item.get("metadata", {})
        if str(metadata.get("record_schema", "")).lower() != "qa_record":
            continue
        question = str(metadata.get("question", "")).strip()
        answer = str(metadata.get("answer", "")).strip()
        if not question or not answer:
            continue

        if _normalize_qa_text(question) == normalized_query:
            best_item = item
            best_answer = answer
            best_score = float("inf")
            break

        match_score = lexical_score(query, question)
        if match_score < 20.0 or match_score <= best_score:
            continue
        best_item = item
        best_answer = answer
        best_score = match_score

    if best_item is None:
        return None

    return {
        "answer": best_answer,
        "citations": [
            {
                "evidence_id": best_item["evidence_id"],
                "source_path": best_item["source_path"],
                "location": best_item["location"],
                "retrieval_source": best_item["retrieval_source"],
            }
        ],
        "answerable": True,
        "answer_support": {
            "explicit": True,
            "reason": "faq_exact_match",
            "max_support_score": 999.0,
            "threshold": _explicit_support_threshold(None),
            "matched_evidence_ids": [best_item["evidence_id"]],
        },
    }


def _normalize_qa_text(value: str) -> str:
    normalized = normalize_text(value)
    for source, target in (
        ("帐户", "账户"),
        ("訂單", "订单"),
        ("訂购", "订购"),
        ("訂單號", "订单号"),
    ):
        normalized = normalized.replace(source, target)
    normalized = "".join(
        char for char in normalized if char.isalnum() or "一" <= char <= "鿿"
    )
    return normalized


def _generation_evidence(
    reranked: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in reranked + merged:
        evidence_id = str(item.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _explicit_answer_support(
    *,
    query: str,
    evidence: list[dict[str, Any]],
    query_plan: QueryConstraintPlan | dict[str, Any] | None,
) -> dict[str, Any]:
    plan = _coerce_query_plan(query, query_plan)
    threshold = _explicit_support_threshold(plan)
    best_score = 0.0
    best_id = ""
    matched_ids: list[str] = []

    for item in evidence:
        score = lexical_score(
            query,
            f"{item.get('source_path', '')}\n{item.get('content', '')}",
            query_plan=plan,
        )
        if score > best_score:
            best_score = float(score)
            best_id = str(item.get("evidence_id", ""))
        if score >= threshold:
            evidence_id = str(item.get("evidence_id", ""))
            if evidence_id:
                matched_ids.append(evidence_id)

    return {
        "explicit": best_score >= threshold,
        "reason": "lexical_grounding",
        "max_support_score": float(best_score),
        "threshold": float(threshold),
        "matched_evidence_ids": matched_ids[:5] if matched_ids else ([best_id] if best_id and best_score > 0 else []),
    }


def _explicit_support_threshold(plan: QueryConstraintPlan | None) -> float:
    if plan is None:
        return 6.0
    if plan.hard_terms:
        return 7.5
    if plan.answer_shape in {"count", "comparison", "table", "list"}:
        return 6.5
    return 6.0


def _coerce_query_plan(
    query: str,
    query_plan: QueryConstraintPlan | dict[str, Any] | None,
) -> QueryConstraintPlan | None:
    if query_plan is None:
        return None
    if isinstance(query_plan, QueryConstraintPlan):
        return query_plan
    return query_plan_from_dict({"raw_query": query, **query_plan})
