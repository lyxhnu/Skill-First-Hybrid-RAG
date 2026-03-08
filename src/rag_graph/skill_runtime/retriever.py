from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..config import Settings
from ..types import QueryConstraintPlan, query_plan_from_dict
from ..utils.io import iter_jsonl, read_text_with_fallback
from ..utils.text import lexical_score


class ChunkRepository:
    def __init__(self, chunks_path: Path):
        self.chunks_path = chunks_path
        self._chunks: list[dict[str, Any]] = []

    def reload(self) -> None:
        self._chunks = list(iter_jsonl(self.chunks_path))

    @property
    def chunks(self) -> list[dict[str, Any]]:
        return self._chunks


class SkillRetriever:
    def __init__(self, settings: Settings, repository: ChunkRepository):
        self.settings = settings
        self.repository = repository
        self.pdf_ref = settings.project_root / ".agent" / "skills" / "rag-skill" / "references" / "pdf_reading.md"
        self.excel_read_ref = (
            settings.project_root / ".agent" / "skills" / "rag-skill" / "references" / "excel_reading.md"
        )
        self.excel_analysis_ref = (
            settings.project_root
            / ".agent"
            / "skills"
            / "rag-skill"
            / "references"
            / "excel_analysis.md"
        )
        self._reference_cache: dict[str, dict[str, Any]] = {}

    def retrieve(
        self,
        query: str,
        candidate_files: list[str],
        top_k: int,
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        allowed = set(candidate_files)
        reference_metadata = self._load_required_reference_metadata(allowed)
        plan = _coerce_query_plan(query, query_plan)

        scored: list[dict[str, Any]] = []
        for chunk in self.repository.chunks:
            source_path = chunk["source_path"]
            if allowed and source_path not in allowed:
                continue
            target_payload = f"{Path(source_path).name}\n{chunk['content']}"
            score = lexical_score(query, target_payload, query_plan=plan)
            if score <= 0:
                continue
            payload = dict(chunk)
            payload["retrieval_source"] = "skill:rag-skill"
            payload["score"] = float(score)
            payload["metadata"] = self._merge_reference_metadata(
                chunk.get("metadata", {}),
                source_path=source_path,
                reference_metadata=reference_metadata,
            )
            scored.append(payload)

        scored.sort(key=lambda item: item["score"], reverse=True)
        ranked = scored[:top_k]
        ranked = self._augment_pdf_neighbors(
            query,
            ranked,
            allowed,
            plan,
            reference_metadata,
            limit=max(top_k * 2, top_k),
        )
        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[:top_k]

    def _load_required_reference_metadata(self, candidate_files: set[str]) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        if not candidate_files:
            return metadata
        has_pdf = any(path.lower().endswith(".pdf") for path in candidate_files)
        has_excel = any(path.lower().endswith(".xlsx") for path in candidate_files)
        if has_pdf:
            metadata["pdf"] = self._load_reference_metadata("pdf", [self.pdf_ref])
        if has_excel:
            metadata["xlsx"] = self._load_reference_metadata("xlsx", [self.excel_read_ref, self.excel_analysis_ref])
        return metadata

    def _load_reference_metadata(self, cache_key: str, required: list[Path]) -> dict[str, Any]:
        cached = self._reference_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Missing required rag-skill references: {', '.join(missing)}")

        reference_hashes: dict[str, str] = {}
        for path in required:
            content = read_text_with_fallback(path)
            reference_hashes[path.name] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        payload = {
            "references_loaded": [str(path.resolve()) for path in required],
            "reference_hashes": reference_hashes,
        }
        self._reference_cache[cache_key] = payload
        return dict(payload)

    @staticmethod
    def _merge_reference_metadata(
        existing: dict[str, Any] | None,
        *,
        source_path: str,
        reference_metadata: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = dict(existing or {})
        file_type = Path(source_path).suffix.lower().lstrip(".")
        selected = reference_metadata.get(file_type)
        if not selected:
            return metadata
        loaded = list(metadata.get("references_loaded", []))
        for path in selected.get("references_loaded", []):
            if path not in loaded:
                loaded.append(path)
        reference_hashes = dict(metadata.get("reference_hashes", {}))
        reference_hashes.update(selected.get("reference_hashes", {}))
        metadata["references_loaded"] = loaded
        metadata["reference_hashes"] = reference_hashes
        return metadata

    def _augment_pdf_neighbors(
        self,
        query: str,
        ranked: list[dict[str, Any]],
        allowed: set[str],
        plan: QueryConstraintPlan | None,
        reference_metadata: dict[str, dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not ranked:
            return []

        merged = {row["evidence_id"]: dict(row) for row in ranked}
        for row in list(ranked):
            source_path = str(row.get("source_path", ""))
            if not source_path.lower().endswith(".pdf"):
                continue
            location = row.get("location", {})
            if not isinstance(location, dict) or "page" not in location:
                continue
            try:
                page = int(location["page"])
            except Exception:
                continue
            for neighbor_page in (page - 1, page + 1):
                if neighbor_page < 1:
                    continue
                neighbor = self._find_pdf_page(source_path, neighbor_page, allowed)
                if neighbor is None or neighbor["evidence_id"] in merged:
                    continue
                neighbor_payload = dict(neighbor)
                neighbor_payload["retrieval_source"] = "skill:rag-skill"
                lexical = lexical_score(
                    query,
                    f"{Path(source_path).name}\n{neighbor_payload['content']}",
                    query_plan=plan,
                )
                neighbor_payload["score"] = max(float(row.get("score", 0.0)) * 0.9, float(lexical))
                neighbor_payload["metadata"] = self._merge_reference_metadata(
                    neighbor.get("metadata", {}),
                    source_path=source_path,
                    reference_metadata=reference_metadata,
                )
                merged[neighbor_payload["evidence_id"]] = neighbor_payload
                if len(merged) >= limit:
                    return list(merged.values())
        return list(merged.values())

    def _find_pdf_page(
        self,
        source_path: str,
        page: int,
        allowed: set[str],
    ) -> dict[str, Any] | None:
        for chunk in self.repository.chunks:
            if allowed and chunk["source_path"] not in allowed:
                continue
            if chunk["source_path"] != source_path:
                continue
            if not str(chunk["source_path"]).lower().endswith(".pdf"):
                continue
            location = chunk.get("location", {})
            if isinstance(location, dict) and int(location.get("page", -1)) == page:
                return dict(chunk)
        return None


def _coerce_query_plan(
    query: str, query_plan: QueryConstraintPlan | dict[str, Any] | None
) -> QueryConstraintPlan | None:
    if query_plan is None:
        return None
    if isinstance(query_plan, QueryConstraintPlan):
        return query_plan
    return query_plan_from_dict({"raw_query": query, **query_plan})
