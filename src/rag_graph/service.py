from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from .config import Settings
from .eval.evaluator import evaluate_service
from .feedback.manager import CustomerServiceFeedbackManager
from .fusion.fuse import EvidenceFusion
from .graph.workflow import build_workflow
from .memory.manager import MemoryManager
from .models.providers import EmbeddingGateway, ModelGateway, RerankGateway
from .parser_cache.ingest import IngestEngine
from .query_runtime.analyzer import QueryConstraintAnalyzer
from .skill_runtime.manager import SkillManager
from .skill_runtime.registry import SkillRegistry
from .skill_runtime.retriever import ChunkRepository, SkillRetriever
from .skill_runtime.router import SkillRouter
from .vector_store.index import VectorStore


class RAGService:
    def __init__(self, settings: Settings | None = None):
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.pid = os.getpid()
        self.settings = settings or Settings()
        self.settings.ensure_storage_dirs()

        self.ingest_engine = IngestEngine(self.settings)
        self.chunk_repo = ChunkRepository(self.settings.parsed_chunks_path)
        self.chunk_repo.reload()

        self.skill_registry = SkillRegistry(self.settings)
        self.embedding_gateway = EmbeddingGateway(self.settings)
        self.router = SkillRouter(self.settings, embedding_gateway=self.embedding_gateway)
        self.skill_retriever = SkillRetriever(self.settings, self.chunk_repo)
        self.query_analyzer = QueryConstraintAnalyzer(self.settings)
        self.skill_manager = SkillManager(
            settings=self.settings,
            registry=self.skill_registry,
            router=self.router,
            retriever=self.skill_retriever,
        )
        self.memory_manager = MemoryManager(self.settings)
        self.vector_store = VectorStore(self.settings, self.embedding_gateway)
        self.fusion = EvidenceFusion(self.settings)
        self.model_gateway = ModelGateway(self.settings)
        self.rerank_gateway = RerankGateway(self.settings)
        self.feedback_manager = CustomerServiceFeedbackManager(self.settings)
        self.workflow = build_workflow(
            settings=self.settings,
            router=self.router,
            skill_manager=self.skill_manager,
            vector_store=self.vector_store,
            fusion=self.fusion,
            model_gateway=self.model_gateway,
            rerank_gateway=self.rerank_gateway,
            query_analyzer=self.query_analyzer,
            memory_manager=self.memory_manager,
        )

    def ingest(self, force: bool = False) -> dict[str, Any]:
        manifest = self.ingest_engine.ingest(force=force)
        self.chunk_repo.reload()
        self.router.reload()
        index_summary = self.vector_store.build(self.chunk_repo.chunks)
        return {"manifest": manifest, "vector_index": index_summary}

    def query(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        *,
        session_id: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        mode = mode.lower().strip()
        if mode not in {"skill", "hybrid", "vector"}:
            raise ValueError("mode must be one of: skill, hybrid, vector")
        top_k = top_k or self.settings.default_top_k
        if top_k < 1:
            top_k = 1
        if top_k > self.settings.max_top_k:
            top_k = self.settings.max_top_k

        state_input = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "session_id": session_id or "default-session",
            "actor_id": actor_id or "default-user",
        }
        state_output = self.workflow.invoke(state_input)
        evidence_trace = {
            "structured_hits": sum(
                1
                for item in state_output.get("skill_evidence", [])
                if str(item.get("retrieval_source", "")).startswith("skill:rag-skill:excel-analysis")
            ),
            "skill_hits": len(state_output.get("skill_evidence", [])),
            "vector_hits": len(state_output.get("vector_evidence", [])),
            "merged_hits": len(state_output.get("merged_evidence", [])),
            "reranked_hits": len(state_output.get("reranked_evidence", [])),
        }
        answer_support = dict(state_output.get("answer_support", {}))
        knowledge_found = bool(answer_support.get("explicit", False))
        feedback_capture = self.feedback_manager.capture_gap(
            query=query,
            effective_query=str(state_output.get("effective_query", query)),
            knowledge_found=knowledge_found,
            session_id=str(state_output.get("session_id", session_id or "default-session")),
            actor_id=str(state_output.get("actor_id", actor_id or "default-user")),
            mode=mode,
            confidence=float(state_output.get("confidence", 0.0) or 0.0),
            selected_skills=list(state_output.get("selected_skills", [])),
            candidate_dirs=list(state_output.get("candidate_dirs", [])),
            candidate_files=list(state_output.get("candidate_files", [])),
            evidence_trace=evidence_trace,
        )
        evidence_preview = _build_evidence_preview(state_output)
        return {
            "query": query,
            "mode": mode,
            "session_id": state_output.get("session_id", session_id or "default-session"),
            "actor_id": state_output.get("actor_id", actor_id or "default-user"),
            "effective_query": state_output.get("effective_query", query),
            "answer": state_output.get("answer", ""),
            "answerable": bool(state_output.get("answerable", False)),
            "citations": state_output.get("citations", []),
            "query_constraints": state_output.get("query_constraints", {}),
            "memory_trace": state_output.get("memory_trace", {}),
            "selected_skills": state_output.get("selected_skills", []),
            "candidate_dirs": state_output.get("candidate_dirs", []),
            "candidate_files": state_output.get("candidate_files", []),
            "confidence": state_output.get("confidence", 0.0),
            "evidence_trace": evidence_trace,
            "answer_support": answer_support,
            "evidence_preview": evidence_preview,
            "selected_models": state_output.get("selected_models", {}),
            "customer_service_feedback": feedback_capture or {"captured": False},
        }

    def evaluate(self, mode: str = "hybrid") -> dict[str, Any]:
        return evaluate_service(self, mode=mode)

    def list_skills(self) -> list[dict[str, Any]]:
        self.skill_registry.reload()
        return self.skill_manager.list_skills()

    def execute_skill(self, skill_id: str, query: str, top_k: int = 8) -> dict[str, Any]:
        query_plan = self.query_analyzer.analyze(query)
        candidate_files: list[str] | None = None
        if skill_id == "rag-skill":
            _, candidate_files = self.router.route(query, query_plan=query_plan, max_domains=2, max_files=12)
            query_plan = self.query_analyzer.refine_for_files(query, query_plan, candidate_files)
        evidence = self.skill_manager.execute_skill_retrieval(
            skill_id=skill_id,
            query=query,
            top_k=top_k,
            candidate_files=candidate_files,
            query_plan=query_plan,
        )
        reranked = self.rerank_gateway.rerank(query, evidence, top_k=top_k, query_plan=query_plan)
        if reranked:
            answer = self.model_gateway.generate(query, reranked, query_plan=query_plan)
            answerable = True
            citations = [
                {
                    "evidence_id": item["evidence_id"],
                    "source_path": item["source_path"],
                    "location": item["location"],
                    "retrieval_source": item["retrieval_source"],
                }
                for item in reranked[:5]
            ]
        else:
            answer = "未在该技能的可用知识中检索到相关内容，无法回答该问题。"
            answerable = False
            citations = []
        return {
            "skill_id": skill_id,
            "query": query,
            "answer": answer,
            "answerable": answerable,
            "evidence_count": len(reranked),
            "citations": citations,
            "evidence": reranked[:top_k],
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "knowledge_dir_exists": self.settings.knowledge_dir.exists(),
            "chunks_loaded": len(self.chunk_repo.chunks),
            "vector_backend": self.vector_store.backend,
            "vector_index_ready": self.vector_store.ready,
            "vector_index_metadata": dict(self.vector_store.metadata),
            "chat_provider": self.settings.chat_provider,
            "chat_model": self.settings.chat_model,
            "embed_provider": self.settings.embed_provider,
            "embed_model": self.settings.embed_model,
            "rerank_provider": self.settings.rerank_provider,
            "rerank_model": self.settings.rerank_model,
            "vector_dimension": self.vector_store.dimension,
            "registered_skills": len(self.skill_registry.list_skills()),
            "customer_service_feedback": self.feedback_manager.stats(),
            "pid": self.pid,
            "startup_time": self.started_at,
            "project_root": str(self.settings.project_root),
            "service_file": __file__,
            "vector_index_path": str(self.vector_store.index_path),
            "memory_dir": str(self.settings.memory_dir),
        }

    def list_customer_service_gaps(self, *, status: str = "open", limit: int = 100) -> dict[str, Any]:
        return self.feedback_manager.list_gaps(status=status, limit=limit)

    def resolve_customer_service_gap(
        self,
        *,
        gap_id: str,
        answer: str,
        reviewer: str | None = None,
        label: str | None = None,
        question: str | None = None,
        url: str | None = None,
        auto_ingest: bool = True,
    ) -> dict[str, Any]:
        payload = self.feedback_manager.resolve_gap(
            gap_id=gap_id,
            answer=answer,
            reviewer=reviewer,
            label=label,
            question=question,
            url=url,
        )
        if auto_ingest:
            payload["ingest"] = self.ingest(force=False)
        return payload


def _build_evidence_preview(state_output: dict[str, Any]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(state_output.get("reranked_evidence", [])) + list(state_output.get("merged_evidence", [])):
        evidence_id = str(item.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        preview.append(
            {
                "evidence_id": evidence_id,
                "source_path": str(item.get("source_path", "")),
                "location": dict(item.get("location", {})),
                "retrieval_source": str(item.get("retrieval_source", "")),
                "score": float(item.get("score", 0.0) or 0.0),
                "file_type": str(item.get("file_type", "")),
                "domain": str(item.get("domain", "")),
                "snippet": str(item.get("content", "")).replace("\n", " ")[:220],
            }
        )
        if len(preview) >= 5:
            break
    return preview
