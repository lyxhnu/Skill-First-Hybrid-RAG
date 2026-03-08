from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import Settings
from ..models.providers import EmbeddingGateway
from ..types import QueryConstraintPlan, query_plan_from_dict
from ..utils.io import read_text_with_fallback
from ..utils.text import lexical_score

SUPPORTED_FILE_TYPES = {".md", ".txt", ".pdf", ".xlsx"}


class SkillRouter:
    def __init__(self, settings: Settings, embedding_gateway: EmbeddingGateway | None = None):
        self.settings = settings
        self.embedding_gateway = embedding_gateway
        self._embedding_cache: dict[str, np.ndarray] = {}

    def reload(self) -> None:
        self._embedding_cache = {}

    def route(
        self,
        query: str,
        query_plan: QueryConstraintPlan | dict[str, object] | None = None,
        max_domains: int = 2,
        max_files: int = 10,
    ) -> tuple[list[str], list[str]]:
        plan = _coerce_query_plan(query, query_plan)
        domains = self._domain_candidates(query, plan)
        candidate_domains = domains[:max_domains] if domains else []
        candidate_files = self._file_candidates(query, candidate_domains, max_files=max_files, query_plan=plan)
        return [str(path.resolve()) for path in candidate_domains], [str(path.resolve()) for path in candidate_files]

    def _domain_candidates(self, query: str, query_plan: QueryConstraintPlan | None = None) -> list[Path]:
        root = self.settings.knowledge_dir
        if not root.exists():
            return []
        scored: list[tuple[float, Path]] = []
        payloads: dict[str, str] = {}
        for child in root.iterdir():
            if not child.is_dir():
                continue
            structure_text = self._read_data_structure(child)
            payload = f"{child.name}\n{structure_text}"
            payloads[str(child.resolve())] = payload
            score = lexical_score(query, payload, extra_terms=[child.name], query_plan=query_plan)
            scored.append((score, child))
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > 0:
            positive = [item[1] for item in scored if item[0] > 0]
            return self._rerank_with_semantics(query, positive, payloads)

        semantic = self._semantic_rank(query, [(path, payloads[str(path.resolve())]) for _, path in scored])
        if semantic:
            return semantic
        return [item[1] for item in scored]

    def _file_candidates(
        self,
        query: str,
        domains: Iterable[Path],
        max_files: int,
        query_plan: QueryConstraintPlan | None = None,
    ) -> list[Path]:
        scored: list[tuple[float, Path]] = []
        payloads: dict[str, str] = {}
        for domain in domains:
            domain_structure = self._read_data_structure(domain)
            for file in domain.rglob("*"):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in SUPPORTED_FILE_TYPES:
                    continue
                if file.name == "data_structure.md":
                    continue
                file_payload = f"{domain.name}\n{file.relative_to(domain)}\n{file.name}\n{domain_structure}"
                payloads[str(file.resolve())] = file_payload
                score = lexical_score(
                    query,
                    file_payload,
                    extra_terms=[file.stem, file.name, domain.name],
                    query_plan=query_plan,
                )
                scored.append((score, file))
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > 0:
            filtered = [item[1] for item in scored if item[0] > 0]
            reranked = self._rerank_with_semantics(query, filtered, payloads)
            return reranked[:max_files]

        semantic = self._semantic_rank(query, [(path, payloads[str(path.resolve())]) for _, path in scored])
        if semantic:
            return semantic[:max_files]
        return [item[1] for item in scored[:max_files]]

    @staticmethod
    def _read_data_structure(directory: Path) -> str:
        index_file = directory / "data_structure.md"
        if not index_file.exists():
            return ""
        return read_text_with_fallback(index_file)

    def _rerank_with_semantics(
        self, query: str, paths: list[Path], payloads: dict[str, str]
    ) -> list[Path]:
        if not paths or self.embedding_gateway is None:
            return paths
        semantic = self._semantic_rank(query, [(path, payloads[str(path.resolve())]) for path in paths])
        if not semantic:
            return paths
        semantic_map = {str(path.resolve()): rank for rank, path in enumerate(semantic)}
        return sorted(paths, key=lambda path: semantic_map.get(str(path.resolve()), len(paths)))

    def _semantic_rank(self, query: str, items: list[tuple[Path, str]]) -> list[Path]:
        if not items or self.embedding_gateway is None:
            return []

        try:
            query_vector = self.embedding_gateway.embed_query(query)
            if query_vector.ndim != 1:
                query_vector = query_vector.reshape(-1)
            item_vectors = self._embed_payloads([payload for _, payload in items])
            if item_vectors.size == 0 or item_vectors.shape[1] != query_vector.shape[0]:
                return []
            scores = np.matmul(item_vectors, query_vector)
        except Exception:
            return []

        ranked = sorted(
            ((float(scores[idx]), path) for idx, (path, _) in enumerate(items)),
            key=lambda item: item[0],
            reverse=True,
        )
        return [path for score, path in ranked if score > 0]

    def _embed_payloads(self, payloads: list[str]) -> np.ndarray:
        if not payloads:
            return np.zeros((0, 0), dtype=np.float32)

        missing = [payload for payload in payloads if payload not in self._embedding_cache]
        if missing:
            matrix = self.embedding_gateway.embed_documents(missing)
            for payload, vector in zip(missing, matrix, strict=False):
                self._embedding_cache[payload] = vector.astype(np.float32)

        vectors = [self._embedding_cache[payload] for payload in payloads if payload in self._embedding_cache]
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)


def _coerce_query_plan(
    query: str, query_plan: QueryConstraintPlan | dict[str, object] | None
) -> QueryConstraintPlan | None:
    if query_plan is None:
        return None
    if isinstance(query_plan, QueryConstraintPlan):
        return query_plan
    return query_plan_from_dict({"raw_query": query, **query_plan})
