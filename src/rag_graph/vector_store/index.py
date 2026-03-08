from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings
from ..models.providers import EmbeddingGateway


class VectorStore:
    def __init__(self, settings: Settings, embedding_gateway: EmbeddingGateway):
        self.settings = settings
        self.embedding_gateway = embedding_gateway
        self.index_path: Path = settings.vector_index_path
        self.matrix: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.load()

    def build(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        self.records = list(chunks)
        if not self.records:
            self.matrix = None
            self._persist()
            return {"indexed_chunks": 0}

        texts = [row["content"] for row in self.records]
        matrix = self.embedding_gateway.embed_documents(texts)
        if matrix.ndim != 2:
            raise RuntimeError("Embedding provider returned invalid matrix shape")
        self.matrix = matrix.astype(np.float32)
        self.metadata = {
            "provider": self.settings.embed_provider,
            "model": self._active_model_name(),
            "dimension": int(self.matrix.shape[1]),
        }
        self._persist()
        return {"indexed_chunks": len(self.records), "dimension": int(self.matrix.shape[1])}

    def search(self, query: str, top_k: int, allowed_files: set[str] | None = None) -> list[dict[str, Any]]:
        if self.matrix is None or not self.records:
            return []
        query_vector = self.embedding_gateway.embed_query(query)
        if query_vector.ndim != 1:
            query_vector = query_vector.reshape(-1)
        if self.matrix.ndim != 2 or self.matrix.shape[1] != query_vector.shape[0]:
            self.matrix = None
            self.records = []
            self.metadata = {}
            return []
        scores = np.matmul(self.matrix, query_vector)
        ranked_indices = np.argsort(scores)[::-1]

        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(scores[int(idx)])
            if score <= 0:
                continue
            record = self.records[int(idx)]
            if allowed_files and record["source_path"] not in allowed_files:
                continue
            payload = dict(record)
            payload["retrieval_source"] = "vector"
            payload["score"] = score
            results.append(payload)
            if len(results) >= top_k:
                break
        return results

    def load(self) -> None:
        if not self.index_path.exists():
            return
        with self.index_path.open("rb") as file:
            payload = pickle.load(file)
        self.matrix = payload.get("matrix")
        self.records = payload.get("records", [])
        self.metadata = payload.get("metadata", {})
        if self.matrix is not None and not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.float32)
        if self.metadata:
            expected_provider = self.settings.embed_provider
            expected_model = self._active_model_name()
            if (
                self.metadata.get("provider") != expected_provider
                or self.metadata.get("model") != expected_model
            ):
                self.matrix = None
                self.records = []
                self.metadata = {}

    def _persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"matrix": self.matrix, "records": self.records, "metadata": self.metadata}
        with self.index_path.open("wb") as file:
            pickle.dump(payload, file)

    def _active_model_name(self) -> str:
        provider = self.settings.embed_provider.lower().strip()
        if provider in {"openai", "openai-compatible"}:
            return self.settings.openai_embedding_model
        if provider == "bailian":
            return self.settings.bailian_embedding_model
        if provider == "zhipu":
            return self.settings.zhipu_embedding_model
        if provider == "gemini":
            return self.settings.gemini_embedding_model
        return self.settings.embed_model
