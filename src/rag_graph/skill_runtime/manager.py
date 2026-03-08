from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..config import Settings
from ..types import QueryConstraintPlan, query_plan_from_dict
from ..utils.io import read_text_with_fallback
from ..utils.text import chunk_text, lexical_score
from .excel_analyzer import ExcelStructuredAnalyzer
from .registry import SkillRegistry, SkillSpec
from .retriever import SkillRetriever
from .router import SkillRouter


class SkillManager:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: SkillRegistry,
        router: SkillRouter,
        retriever: SkillRetriever,
    ):
        self.settings = settings
        self.registry = registry
        self.router = router
        self.retriever = retriever
        self.excel_analyzer = ExcelStructuredAnalyzer(settings)

    def list_skills(self) -> list[dict[str, Any]]:
        return self.registry.list_skills()

    def select_skills(self, query: str, top_n: int = 3) -> list[str]:
        return self.registry.select_for_query(query, top_n=top_n)

    def retrieve_for_query(
        self,
        *,
        query: str,
        query_plan: QueryConstraintPlan | dict[str, Any] | None,
        selected_skills: list[str],
        candidate_files: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        plan = _coerce_query_plan(query, query_plan)
        if not selected_skills:
            selected_skills = self.registry.select_for_query(query, top_n=2)

        evidence: list[dict[str, Any]] = []
        for skill_id in selected_skills:
            skill_evidence = self.execute_skill_retrieval(
                skill_id=skill_id,
                query=query,
                query_plan=plan,
                candidate_files=candidate_files,
                top_k=top_k,
            )
            evidence.extend(skill_evidence)

        evidence.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return evidence[: max(top_k * 2, top_k)]

    def execute_skill_retrieval(
        self,
        *,
        skill_id: str,
        query: str,
        top_k: int,
        candidate_files: list[str] | None = None,
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        spec = self.registry.get(skill_id)
        if spec is None:
            raise ValueError(f"Unknown skill_id: {skill_id}")
        plan = _coerce_query_plan(query, query_plan)
        if spec.skill_id == "rag-skill":
            files = list(candidate_files or [])
            if not files:
                _, files = self.router.route(query, query_plan=plan, max_domains=2, max_files=12)
            structured_evidence = self.excel_analyzer.analyze(
                query=query,
                candidate_files=files,
                top_k=top_k,
                query_plan=plan,
            )
            evidence = self.retriever.retrieve(query=query, candidate_files=files, top_k=top_k, query_plan=plan)
            for row in evidence:
                row["retrieval_source"] = "skill:rag-skill"
                row.setdefault("metadata", {})
                row["metadata"]["skill_id"] = "rag-skill"
            return self._merge_evidence(structured_evidence, evidence, top_k=max(top_k * 2, top_k))
        return self._retrieve_from_skill_assets(spec, query, top_k, plan)

    def _retrieve_from_skill_assets(
        self,
        spec: SkillSpec,
        query: str,
        top_k: int,
        query_plan: QueryConstraintPlan | None = None,
    ) -> list[dict[str, Any]]:
        assets = [spec.skill_md_path] + spec.references + spec.scripts
        scored: list[dict[str, Any]] = []
        for asset in assets:
            if not asset.exists() or not asset.is_file():
                continue
            text = read_text_with_fallback(asset)
            for start, end, chunk in chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap):
                score = lexical_score(
                    query,
                    f"{asset.name}\n{chunk}",
                    extra_terms=[spec.name, spec.skill_id, asset.name],
                    query_plan=query_plan,
                )
                if score <= 0:
                    continue
                location = _char_offset_to_line_range(text, start, end)
                scored.append(
                    {
                        "evidence_id": _build_evidence_id(asset, location, chunk),
                        "source_path": str(asset),
                        "file_type": asset.suffix.lower().lstrip(".") or "txt",
                        "location": location,
                        "content": chunk,
                        "retrieval_source": f"skill:{spec.skill_id}",
                        "score": float(score),
                        "domain": ".agent/skills",
                        "metadata": {"skill_id": spec.skill_id, "skill_name": spec.name},
                    }
                )

        scored.sort(key=lambda row: row["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _merge_evidence(
        primary: list[dict[str, Any]], secondary: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in primary + secondary:
            existing = merged.get(row["evidence_id"])
            if existing is None or float(row.get("score", 0.0)) > float(existing.get("score", 0.0)):
                merged[row["evidence_id"]] = row
        ranked = sorted(merged.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return ranked[:top_k]


def _coerce_query_plan(
    query: str, query_plan: QueryConstraintPlan | dict[str, Any] | None
) -> QueryConstraintPlan | None:
    if query_plan is None:
        return None
    if isinstance(query_plan, QueryConstraintPlan):
        return query_plan
    return query_plan_from_dict({"raw_query": query, **query_plan})


def _char_offset_to_line_range(text: str, start: int, end: int) -> dict[str, int]:
    line_start = text[:start].count("\n") + 1
    line_end = text[:end].count("\n") + 1
    return {"line_start": line_start, "line_end": max(line_end, line_start)}


def _build_evidence_id(path: Path, location: dict[str, int], content: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(str(location).encode("utf-8"))
    digest.update(content.encode("utf-8"))
    return digest.hexdigest()
