from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings
from ..models.providers import EmbeddingGateway
from ..utils.text import normalize_text

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_TOKEN = re.compile(r"[A-Za-z0-9_]{2,}")


class VectorStore:
    def __init__(self, settings: Settings, embedding_gateway: EmbeddingGateway):
        self.settings = settings
        self.embedding_gateway = embedding_gateway
        configured_backend = settings.vector_backend.lower().strip()
        self.backend = "faiss" if configured_backend != "faiss" else configured_backend
        self.index_path: Path = settings.vector_index_path
        self.matrix: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self._doc_term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._path_to_ids: dict[str, list[int]] = {}
        self._faiss_index: Any | None = None
        self._load_error = ""
        self.load()

    @property
    def ready(self) -> bool:
        return self.matrix is not None and bool(self.records)

    @property
    def dimension(self) -> int:
        if self.matrix is not None and self.matrix.ndim == 2:
            return int(self.matrix.shape[1])
        return int(self.metadata.get("dimension", 0) or 0)

    def build(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        self.records = list(chunks)
        if not self.records:
            self.matrix = None
            self._faiss_index = None
            self._doc_term_freqs = []
            self._doc_lengths = []
            self._doc_freqs = {}
            self._avg_doc_length = 0.0
            self._path_to_ids = {}
            self.metadata = self._build_metadata(indexed_chunks=0, dimension=0, ready=False)
            self._persist()
            return {"backend": self.backend, "indexed_chunks": 0}

        texts = [str(row.get("content", "")) for row in self.records]
        matrix = self.embedding_gateway.embed_documents(texts)
        if matrix.ndim != 2:
            raise RuntimeError("Embedding provider returned invalid matrix shape")
        self.matrix = matrix.astype(np.float32)

        self._build_bm25_index()
        self._build_path_lookup()
        self._rebuild_faiss_index()

        self.metadata = self._build_metadata(
            indexed_chunks=len(self.records),
            dimension=int(self.matrix.shape[1]),
            ready=self.ready,
        )
        self._persist()
        return {
            "backend": self.backend,
            "indexed_chunks": len(self.records),
            "dimension": int(self.matrix.shape[1]),
            "faiss_available": bool(self.metadata.get("faiss_available", False)),
        }

    def search(self, query: str, top_k: int, allowed_files: set[str] | None = None) -> list[dict[str, Any]]:
        if not self.ready:
            return []

        active_query = str(query or "").strip()
        if not active_query:
            return []

        branch_limit = min(
            max(top_k * self.settings.faiss_branch_top_k_factor, self.settings.faiss_branch_top_k_min),
            self.settings.faiss_branch_top_k_max,
        )
        dense_hits = self._dense_search(active_query, branch_limit, allowed_files)
        sparse_hits = self._bm25_search(active_query, branch_limit, allowed_files)
        merged = self._merge_hybrid_hits(dense_hits, sparse_hits)

        results: list[dict[str, Any]] = []
        for doc_id, score_bundle in merged[:top_k]:
            record = dict(self.records[doc_id])
            record["retrieval_source"] = "vector:faiss-bm25"
            record["score"] = float(score_bundle["score"])
            record.setdefault("metadata", {})
            record["metadata"]["vector_backend"] = self.backend
            record["metadata"]["hybrid_ranker"] = self.settings.faiss_hybrid_ranker
            record["metadata"]["dense_score"] = float(score_bundle.get("dense_score", 0.0))
            record["metadata"]["bm25_score"] = float(score_bundle.get("bm25_score", 0.0))
            record["metadata"]["faiss_available"] = bool(self.metadata.get("faiss_available", False))
            results.append(record)
        return results

    def load(self) -> None:
        if not self.index_path.exists():
            return
        with self.index_path.open("rb") as file:
            payload = pickle.load(file)

        self.matrix = payload.get("matrix")
        self.records = payload.get("records", [])
        self.metadata = payload.get("metadata", {})
        self._doc_term_freqs = payload.get("doc_term_freqs", [])
        self._doc_lengths = payload.get("doc_lengths", [])
        self._doc_freqs = payload.get("doc_freqs", {})
        self._avg_doc_length = float(payload.get("avg_doc_length", 0.0) or 0.0)
        self._path_to_ids = payload.get("path_to_ids", {})

        if self.matrix is not None and not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.float32)

        if self.metadata:
            expected_provider = self.settings.embed_provider
            expected_model = _active_embedding_model_name(self.settings)
            if self.metadata.get("provider") != expected_provider or self.metadata.get("model") != expected_model:
                self.matrix = None
                self.records = []
                self.metadata = {}
                self._doc_term_freqs = []
                self._doc_lengths = []
                self._doc_freqs = {}
                self._avg_doc_length = 0.0
                self._path_to_ids = {}
                self._faiss_index = None
                return

        self._rebuild_faiss_index()

    def _persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "matrix": self.matrix,
            "records": self.records,
            "metadata": self.metadata,
            "doc_term_freqs": self._doc_term_freqs,
            "doc_lengths": self._doc_lengths,
            "doc_freqs": self._doc_freqs,
            "avg_doc_length": self._avg_doc_length,
            "path_to_ids": self._path_to_ids,
        }
        with self.index_path.open("wb") as file:
            pickle.dump(payload, file)

    def _build_bm25_index(self) -> None:
        self._doc_term_freqs = []
        self._doc_lengths = []
        self._doc_freqs = {}

        for record in self.records:
            payload = f"{Path(str(record.get('source_path', ''))).name}\n{record.get('content', '')}"
            tokens = _bm25_tokenize(payload)
            freqs = Counter(tokens)
            self._doc_term_freqs.append(dict(freqs))
            doc_length = sum(freqs.values())
            self._doc_lengths.append(doc_length)
            for term in freqs:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths
            else 0.0
        )

    def _build_path_lookup(self) -> None:
        self._path_to_ids = {}
        for idx, record in enumerate(self.records):
            source_path = str(record.get("source_path", ""))
            self._path_to_ids.setdefault(source_path, []).append(idx)

    def _rebuild_faiss_index(self) -> None:
        self._faiss_index = None
        self._load_error = ""
        if self.matrix is None or self.matrix.ndim != 2 or self.matrix.size == 0:
            return
        try:
            import faiss

            index = faiss.IndexFlatIP(int(self.matrix.shape[1]))
            index.add(np.ascontiguousarray(self.matrix.astype(np.float32)))
            self._faiss_index = index
            if self.metadata:
                self.metadata["faiss_available"] = True
                self.metadata.pop("load_error", None)
        except Exception as exc:
            self._load_error = str(exc)
            if self.metadata:
                self.metadata["faiss_available"] = False
                self.metadata["load_error"] = self._load_error

    def _dense_search(
        self,
        query: str,
        limit: int,
        allowed_files: set[str] | None,
    ) -> list[tuple[int, float]]:
        if self.matrix is None or not self.records:
            return []
        query_vector = self.embedding_gateway.embed_query(query)
        if query_vector.ndim != 1:
            query_vector = query_vector.reshape(-1)
        query_vector = query_vector.astype(np.float32)

        if allowed_files:
            candidate_ids = self._candidate_ids(allowed_files)
            if not candidate_ids:
                return []
            candidate_matrix = self.matrix[candidate_ids]
            scores = np.matmul(candidate_matrix, query_vector)
            ranked = np.argsort(scores)[::-1]
            results: list[tuple[int, float]] = []
            for idx in ranked[:limit]:
                score = float(scores[int(idx)])
                if score <= 0:
                    continue
                results.append((candidate_ids[int(idx)], score))
            return results

        if self._faiss_index is not None:
            search_limit = min(limit, len(self.records))
            scores, indices = self._faiss_index.search(query_vector.reshape(1, -1), search_limit)
            results = []
            for doc_id, score in zip(indices[0], scores[0], strict=False):
                doc_id = int(doc_id)
                if doc_id < 0:
                    continue
                score = float(score)
                if score <= 0:
                    continue
                results.append((doc_id, score))
            return results

        scores = np.matmul(self.matrix, query_vector)
        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked[:limit]:
            score = float(scores[int(idx)])
            if score <= 0:
                continue
            results.append((int(idx), score))
        return results

    def _bm25_search(
        self,
        query: str,
        limit: int,
        allowed_files: set[str] | None,
    ) -> list[tuple[int, float]]:
        if not self.records or not self._doc_term_freqs:
            return []

        query_terms = _bm25_tokenize(query)
        if not query_terms:
            return []

        candidate_ids = self._candidate_ids(allowed_files) if allowed_files else list(range(len(self.records)))
        scores: list[tuple[int, float]] = []
        total_docs = len(self.records)
        avg_doc_length = self._avg_doc_length or 1.0
        k1 = max(0.1, float(self.settings.bm25_k1))
        b = max(0.0, min(1.0, float(self.settings.bm25_b)))

        for doc_id in candidate_ids:
            term_freqs = self._doc_term_freqs[doc_id]
            doc_length = self._doc_lengths[doc_id] if doc_id < len(self._doc_lengths) else 0
            score = 0.0
            for term in query_terms:
                tf = int(term_freqs.get(term, 0))
                if tf <= 0:
                    continue
                df = int(self._doc_freqs.get(term, 0))
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                norm = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length))
                score += idf * ((tf * (k1 + 1.0)) / max(norm, 1e-12))
            if score > 0:
                scores.append((doc_id, float(score)))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:limit]

    def _merge_hybrid_hits(
        self,
        dense_hits: list[tuple[int, float]],
        sparse_hits: list[tuple[int, float]],
    ) -> list[tuple[int, dict[str, float]]]:
        merged: dict[int, dict[str, float]] = {}
        dense_map = {doc_id: score for doc_id, score in dense_hits}
        sparse_map = {doc_id: score for doc_id, score in sparse_hits}

        if self.settings.faiss_hybrid_ranker.lower().strip() == "rrf":
            k = max(1, int(self.settings.faiss_rrf_k))
            for rank, (doc_id, score) in enumerate(dense_hits, start=1):
                payload = merged.setdefault(doc_id, {"dense_score": 0.0, "bm25_score": 0.0, "score": 0.0})
                payload["dense_score"] = float(score)
                payload["score"] += 1.0 / (k + rank)
            for rank, (doc_id, score) in enumerate(sparse_hits, start=1):
                payload = merged.setdefault(doc_id, {"dense_score": 0.0, "bm25_score": 0.0, "score": 0.0})
                payload["bm25_score"] = float(score)
                payload["score"] += 1.0 / (k + rank)
        else:
            dense_norm = _normalize_scores(dense_map)
            sparse_norm = _normalize_scores(sparse_map)
            dense_weight = float(self.settings.faiss_dense_weight)
            sparse_weight = float(self.settings.faiss_sparse_weight)
            for doc_id in set(dense_map) | set(sparse_map):
                merged[doc_id] = {
                    "dense_score": float(dense_map.get(doc_id, 0.0)),
                    "bm25_score": float(sparse_map.get(doc_id, 0.0)),
                    "score": (dense_weight * dense_norm.get(doc_id, 0.0))
                    + (sparse_weight * sparse_norm.get(doc_id, 0.0)),
                }

        ranked = sorted(merged.items(), key=lambda item: item[1]["score"], reverse=True)
        return ranked

    def _candidate_ids(self, allowed_files: set[str] | None) -> list[int]:
        if not allowed_files:
            return list(range(len(self.records)))
        ids: list[int] = []
        for path in sorted(allowed_files):
            ids.extend(self._path_to_ids.get(path, []))
        return ids

    def _build_metadata(self, *, indexed_chunks: int, dimension: int, ready: bool) -> dict[str, Any]:
        metadata = {
            "backend": self.backend,
            "provider": self.settings.embed_provider,
            "model": _active_embedding_model_name(self.settings),
            "dimension": int(dimension),
            "indexed_chunks": int(indexed_chunks),
            "ready": bool(ready),
            "hybrid_ranker": self.settings.faiss_hybrid_ranker,
            "dense_weight": float(self.settings.faiss_dense_weight),
            "sparse_weight": float(self.settings.faiss_sparse_weight),
            "bm25_k1": float(self.settings.bm25_k1),
            "bm25_b": float(self.settings.bm25_b),
        }
        metadata["faiss_available"] = self._faiss_index is not None
        if self._load_error:
            metadata["load_error"] = self._load_error
        return metadata


def _active_embedding_model_name(settings: Settings) -> str:
    provider = settings.embed_provider.lower().strip()
    if provider in {"openai", "openai-compatible"}:
        return settings.openai_embedding_model
    if provider == "bailian":
        return settings.bailian_embedding_model
    if provider == "zhipu":
        return settings.zhipu_embedding_model
    if provider == "gemini":
        return settings.gemini_embedding_model
    return settings.embed_model


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return {doc_id: 1.0 for doc_id in scores}
    span = maximum - minimum
    return {doc_id: (score - minimum) / span for doc_id, score in scores.items()}


def _bm25_tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    tokens: list[str] = []
    for token in _LATIN_TOKEN.findall(text):
        tokens.append(token)

    for run in _CJK_RUN.findall(text):
        run = run.strip()
        if len(run) <= 2:
            tokens.append(run)
            continue
        max_size = min(4, len(run))
        for size in range(max_size, 1, -1):
            for offset in range(0, len(run) - size + 1):
                tokens.append(run[offset : offset + size])
        if len(run) <= 4:
            tokens.append(run)
    return tokens
