from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Protocol

import numpy as np

from ..config import Settings
from ..types import QueryConstraintPlan, query_plan_from_dict
from ..utils.text import lexical_score


class ChatProvider(Protocol):
    def generate(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
        memory_context: str | None = None,
    ) -> str:
        ...


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


class RerankProvider(Protocol):
    def rerank(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        top_k: int,
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...


class BuiltinExtractiveProvider:
    def generate(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
        memory_context: str | None = None,
    ) -> str:
        if not evidence:
            return "The knowledge base does not contain enough evidence to answer reliably."
        plan = _coerce_query_plan(query, query_plan)
        lines: list[str] = []
        lines.append(f"Question: {query}")
        if plan and plan.hard_terms:
            lines.append(f"Scope: {', '.join(plan.hard_terms)}")
        lines.append("Answer grounded in retrieved evidence:")
        for row in evidence[:3]:
            snippet = row["content"].strip().replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:180] + "..."
            lines.append(f"- {snippet}")
        lines.append("See citations for the exact evidence.")
        return "\n".join(lines)


class OpenAICompatibleChatProvider:
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
        memory_context: str | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OpenAI-compatible API key is not configured")
        client = _build_openai_client(api_key=self.api_key, base_url=self.base_url)
        system_prompt, user_prompt = _build_chat_prompts(query, evidence, query_plan, memory_context=memory_context)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""


class AnthropicChatProvider:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
        memory_context: str | None = None,
    ) -> str:
        if not self.settings.anthropic_api_key:
            raise RuntimeError("RAG_ANTHROPIC_API_KEY is not configured")
        try:
            import anthropic
        except Exception as exc:
            raise RuntimeError("anthropic package is not installed") from exc

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        system_prompt, user_prompt = _build_chat_prompts(query, evidence, query_plan, memory_context=memory_context)
        response = client.messages.create(
            model=self.settings.anthropic_chat_model,
            max_tokens=700,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if not response.content:
            return ""
        return "".join(block.text for block in response.content if hasattr(block, "text"))


class GeminiChatProvider:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
        memory_context: str | None = None,
    ) -> str:
        if not self.settings.gemini_api_key:
            raise RuntimeError("RAG_GEMINI_API_KEY is not configured")
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError("google-generativeai package is not installed") from exc

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(model_name=self.settings.gemini_chat_model)
        system_prompt, user_prompt = _build_chat_prompts(query, evidence, query_plan, memory_context=memory_context)
        response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
        return response.text or ""


class LocalHashEmbeddingProvider:
    _TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]{2,}")

    def __init__(self, settings: Settings):
        self.settings = settings
        self.dim = max(128, int(settings.local_embedding_dim))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = [self._embed_one(text) for text in texts]
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> np.ndarray:
        text = (text or "").lower()
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = self._TOKEN_PATTERN.findall(text)
        if not tokens:
            tokens = [text[i : i + 2] for i in range(max(0, len(text) - 1))]
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            weight = 0.5 + (digest[5] / 255.0)
            vector[index] += sign * weight
        return _normalize_vector(vector)


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.api_key = api_key
        self.model = model
        self.client = _build_openai_client(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError("OpenAI-compatible API key is not configured")
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return np.vstack([_normalize_vector(vec) for vec in vectors])

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return _normalize_vector(np.array(response.data[0].embedding, dtype=np.float32))


class GeminiEmbeddingProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not self.settings.gemini_api_key:
            raise RuntimeError("RAG_GEMINI_API_KEY is not configured")
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError("google-generativeai package is not installed") from exc
        self.genai = genai
        self.genai.configure(api_key=self.settings.gemini_api_key)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = [self._embed_single(text, task_type="retrieval_document") for text in texts]
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_single(text, task_type="retrieval_query")

    def _embed_single(self, text: str, task_type: str) -> np.ndarray:
        result = self.genai.embed_content(
            model=self.settings.gemini_embedding_model,
            content=text or "",
            task_type=task_type,
        )
        embedding = result["embedding"] if isinstance(result, dict) else result.embedding
        return _normalize_vector(np.array(embedding, dtype=np.float32))


class BuiltinLexicalRerankProvider:
    def rerank(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        top_k: int,
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not evidence:
            return []
        scored = list(evidence)
        plan = _coerce_query_plan(query, query_plan)
        raw_scores = [float(item.get("score", 0.0)) for item in scored]
        score_min, score_max = min(raw_scores), max(raw_scores)
        score_span = score_max - score_min if score_max > score_min else 1.0

        lexical_values = [
            math.log1p(
                lexical_score(
                    query,
                    f"{item.get('source_path', '')}\n{item.get('content', '')}",
                    query_plan=plan,
                )
            )
            for item in scored
        ]
        lex_min, lex_max = min(lexical_values), max(lexical_values)
        lex_span = lex_max - lex_min if lex_max > lex_min else 1.0

        reranked: list[dict[str, Any]] = []
        for item, lex in zip(scored, lexical_values):
            base_norm = (float(item.get("score", 0.0)) - score_min) / score_span
            lex_norm = (lex - lex_min) / lex_span
            source_bonus = 0.04 if str(item.get("retrieval_source", "")).startswith("skill") else 0.0
            final_score = (0.65 * base_norm) + (0.35 * lex_norm) + source_bonus
            payload = dict(item)
            payload["score"] = float(final_score)
            payload.setdefault("metadata", {})
            payload["metadata"]["rerank_provider"] = "builtin-lexical"
            reranked.append(payload)

        reranked.sort(key=lambda row: row["score"], reverse=True)
        return reranked[:top_k]


class OpenAICompatibleLLMRerankProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        provider_label: str,
        fallback: RerankProvider,
        base_url: str | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.provider_label = provider_label
        self.fallback = fallback
        self.client = _build_openai_client(api_key=api_key, base_url=base_url)

    def rerank(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        top_k: int,
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not evidence or not self.api_key:
            return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)
        reduced = evidence[: min(12, len(evidence))]
        prompt_items = []
        for item in reduced:
            prompt_items.append(
                {
                    "evidence_id": item["evidence_id"],
                    "source_path": item["source_path"],
                    "location": item.get("location", {}),
                    "text": item["content"][:900],
                }
            )
        try:
            system_prompt, user_prompt = _build_rerank_prompts(query, prompt_items, query_plan)
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = _extract_json_array(content)
            if not parsed:
                return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)
            mapping = {row["evidence_id"]: float(row["score"]) for row in parsed if "evidence_id" in row and "score" in row}
            reranked = []
            for item in evidence:
                payload = dict(item)
                payload["score"] = mapping.get(item["evidence_id"], float(item.get("score", 0.0)))
                payload.setdefault("metadata", {})
                payload["metadata"]["rerank_provider"] = self.provider_label
                reranked.append(payload)
            reranked.sort(key=lambda row: row["score"], reverse=True)
            return reranked[:top_k]
        except Exception:
            return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)


class GeminiLLMRerankProvider:
    def __init__(self, settings: Settings, fallback: RerankProvider):
        self.settings = settings
        self.fallback = fallback

    def rerank(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        top_k: int,
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not evidence or not self.settings.gemini_api_key:
            return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)
        try:
            import google.generativeai as genai
        except Exception:
            return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(model_name=self.settings.gemini_chat_model)
        reduced = evidence[: min(12, len(evidence))]
        prompt_items = []
        for item in reduced:
            prompt_items.append(
                {
                    "evidence_id": item["evidence_id"],
                    "source_path": item["source_path"],
                    "location": item.get("location", {}),
                    "text": item["content"][:900],
                }
            )
        system_prompt, user_prompt = _build_rerank_prompts(query, prompt_items, query_plan)
        try:
            response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            content = response.text or ""
            parsed = _extract_json_array(content)
            if not parsed:
                return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)
            mapping = {row["evidence_id"]: float(row["score"]) for row in parsed if "evidence_id" in row and "score" in row}
            reranked = []
            for item in evidence:
                payload = dict(item)
                payload["score"] = mapping.get(item["evidence_id"], float(item.get("score", 0.0)))
                payload.setdefault("metadata", {})
                payload["metadata"]["rerank_provider"] = "gemini-llm"
                reranked.append(payload)
            reranked.sort(key=lambda row: row["score"], reverse=True)
            return reranked[:top_k]
        except Exception:
            return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)


class DashscopeRerankProvider:
    def __init__(self, settings: Settings, fallback: RerankProvider):
        self.settings = settings
        self.fallback = fallback

    def rerank(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        top_k: int,
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not evidence or not self.settings.bailian_api_key:
            return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)
        try:
            import requests
        except Exception:
            return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)

        max_docs = min(40, len(evidence))
        candidate = evidence[:max_docs]
        documents = [
            f"source={item.get('source_path', '')}\nlocation={item.get('location', {})}\ntext={item.get('content', '')}"
            for item in candidate
        ]
        payload = {
            "model": self.settings.bailian_rerank_model,
            "input": {"query": _augment_rerank_query(query, query_plan), "documents": documents},
            "top_n": min(top_k, len(documents)),
        }
        headers = {
            "Authorization": f"Bearer {self.settings.bailian_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.settings.bailian_rerank_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code >= 300:
                return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)
            body = response.json()
        except Exception:
            return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)

        results = []
        if isinstance(body, dict):
            output = body.get("output")
            if isinstance(output, dict):
                results = output.get("results") or []
            if not results:
                results = body.get("results") or body.get("data") or []

        mapping: dict[int, float] = {}
        if isinstance(results, list):
            for row in results:
                if not isinstance(row, dict):
                    continue
                idx = row.get("index")
                if idx is None and row.get("document_id") is not None:
                    try:
                        idx = int(row["document_id"])
                    except Exception:
                        continue
                if idx is None:
                    continue
                try:
                    score = float(
                        row.get("relevance_score", row.get("score", row.get("relevance", 0.0)))
                    )
                except Exception:
                    score = 0.0
                mapping[int(idx)] = score

        if not mapping:
            return self.fallback.rerank(query, evidence, top_k, query_plan=query_plan)

        reranked: list[dict[str, Any]] = []
        for idx, item in enumerate(candidate):
            payload_item = dict(item)
            payload_item["score"] = mapping.get(idx, float(item.get("score", 0.0)))
            payload_item.setdefault("metadata", {})
            payload_item["metadata"]["rerank_provider"] = "bailian-rerank"
            reranked.append(payload_item)

        if len(evidence) > len(candidate):
            reranked.extend(evidence[len(candidate) :])

        reranked.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return reranked[:top_k]


class ModelGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.builtin = BuiltinExtractiveProvider()

    def generate(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
        memory_context: str | None = None,
    ) -> str:
        provider = self.settings.chat_provider.lower().strip()
        try:
            if provider in {"openai", "openai-compatible"}:
                return OpenAICompatibleChatProvider(
                    api_key=self.settings.openai_api_key or "",
                    base_url=self.settings.openai_base_url,
                    model=self.settings.openai_chat_model,
                ).generate(query, evidence, query_plan=query_plan, memory_context=memory_context)
            if provider == "bailian":
                return OpenAICompatibleChatProvider(
                    api_key=self.settings.bailian_api_key or "",
                    base_url=self.settings.bailian_base_url,
                    model=self.settings.bailian_chat_model,
                ).generate(query, evidence, query_plan=query_plan, memory_context=memory_context)
            if provider == "zhipu":
                return OpenAICompatibleChatProvider(
                    api_key=self.settings.zhipu_api_key or "",
                    base_url=self.settings.zhipu_base_url,
                    model=self.settings.zhipu_chat_model,
                ).generate(query, evidence, query_plan=query_plan, memory_context=memory_context)
            if provider == "anthropic":
                return AnthropicChatProvider(self.settings).generate(
                    query, evidence, query_plan=query_plan, memory_context=memory_context
                )
            if provider == "gemini":
                return GeminiChatProvider(self.settings).generate(
                    query, evidence, query_plan=query_plan, memory_context=memory_context
                )
            return self.builtin.generate(query, evidence, query_plan=query_plan, memory_context=memory_context)
        except Exception:
            return self.builtin.generate(query, evidence, query_plan=query_plan, memory_context=memory_context)


class EmbeddingGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._fallback = LocalHashEmbeddingProvider(settings)
        self._provider_instance: EmbeddingProvider | None = None
        self._runtime_provider = "local-hash"
        self._runtime_model = f"local-hash-{self._fallback.dim}"
        self._runtime_is_fallback = True

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        provider = self._provider()
        try:
            vectors = provider.embed_documents(texts)
            self._mark_runtime(provider)
            return vectors
        except Exception:
            self._provider_instance = self._fallback
            self._mark_fallback()
            return self._fallback.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        provider = self._provider()
        try:
            vector = provider.embed_query(text)
            self._mark_runtime(provider)
            return vector
        except Exception:
            self._provider_instance = self._fallback
            self._mark_fallback()
            return self._fallback.embed_query(text)

    def _provider(self) -> EmbeddingProvider:
        if self._provider_instance is not None:
            return self._provider_instance
        provider = self.settings.embed_provider.lower().strip()
        try:
            if provider in {"openai", "openai-compatible"}:
                self._provider_instance = OpenAICompatibleEmbeddingProvider(
                    api_key=self.settings.openai_api_key or "",
                    base_url=self.settings.openai_base_url,
                    model=self.settings.openai_embedding_model,
                )
                return self._provider_instance
            if provider == "bailian":
                self._provider_instance = OpenAICompatibleEmbeddingProvider(
                    api_key=self.settings.bailian_api_key or "",
                    base_url=self.settings.bailian_base_url,
                    model=self.settings.bailian_embedding_model,
                )
                return self._provider_instance
            if provider == "zhipu":
                self._provider_instance = OpenAICompatibleEmbeddingProvider(
                    api_key=self.settings.zhipu_api_key or "",
                    base_url=self.settings.zhipu_base_url,
                    model=self.settings.zhipu_embedding_model,
                )
                return self._provider_instance
            if provider == "gemini":
                self._provider_instance = GeminiEmbeddingProvider(self.settings)
                return self._provider_instance
            self._provider_instance = self._fallback
            self._mark_fallback()
            return self._provider_instance
        except Exception:
            self._provider_instance = self._fallback
            self._mark_fallback()
            return self._provider_instance

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": self._runtime_provider,
            "model": self._runtime_model,
            "is_fallback": self._runtime_is_fallback,
        }

    def _mark_runtime(self, provider: EmbeddingProvider) -> None:
        if provider is self._fallback:
            self._mark_fallback()
            return
        label = self.settings.embed_provider.lower().strip()
        self._runtime_provider = label or "unknown"
        self._runtime_model = _active_embedding_model_name(self.settings)
        self._runtime_is_fallback = False

    def _mark_fallback(self) -> None:
        self._runtime_provider = "local-hash"
        self._runtime_model = f"local-hash-{self._fallback.dim}"
        self._runtime_is_fallback = True


class RerankGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.builtin = BuiltinLexicalRerankProvider()

    def _prompt_fallback(self) -> RerankProvider:
        if self.settings.zhipu_api_key:
            return OpenAICompatibleLLMRerankProvider(
                api_key=self.settings.zhipu_api_key or "",
                base_url=self.settings.zhipu_base_url,
                model=self.settings.zhipu_chat_model,
                provider_label="zhipu-llm-fallback",
                fallback=self.builtin,
            )
        if self.settings.openai_api_key:
            return OpenAICompatibleLLMRerankProvider(
                api_key=self.settings.openai_api_key or "",
                base_url=self.settings.openai_base_url,
                model=self.settings.openai_chat_model,
                provider_label="openai-llm-fallback",
                fallback=self.builtin,
            )
        if self.settings.gemini_api_key:
            return GeminiLLMRerankProvider(self.settings, self.builtin)
        if self.settings.bailian_api_key and self.settings.bailian_chat_model:
            return OpenAICompatibleLLMRerankProvider(
                api_key=self.settings.bailian_api_key or "",
                base_url=self.settings.bailian_base_url,
                model=self.settings.bailian_chat_model,
                provider_label="bailian-llm-fallback",
                fallback=self.builtin,
            )
        return self.builtin

    def rerank(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        top_k: int,
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        provider = self.settings.rerank_provider.lower().strip()
        if provider in {"openai", "openai-compatible"}:
            return OpenAICompatibleLLMRerankProvider(
                api_key=self.settings.openai_api_key or "",
                base_url=self.settings.openai_base_url,
                model=self.settings.openai_chat_model,
                provider_label="openai-llm",
                fallback=self.builtin,
            ).rerank(query, evidence, top_k, query_plan=query_plan)
        if provider == "bailian":
            return DashscopeRerankProvider(self.settings, self._prompt_fallback()).rerank(
                query, evidence, top_k, query_plan=query_plan
            )
        if provider == "zhipu":
            return OpenAICompatibleLLMRerankProvider(
                api_key=self.settings.zhipu_api_key or "",
                base_url=self.settings.zhipu_base_url,
                model=self.settings.zhipu_chat_model,
                provider_label="zhipu-llm",
                fallback=self.builtin,
            ).rerank(query, evidence, top_k, query_plan=query_plan)
        if provider == "gemini":
            return GeminiLLMRerankProvider(self.settings, self.builtin).rerank(
                query, evidence, top_k, query_plan=query_plan
            )
        return self.builtin.rerank(query, evidence, top_k, query_plan=query_plan)


def _build_openai_client(*, api_key: str, base_url: str | None = None):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is not installed") from exc
    return OpenAI(api_key=api_key, base_url=base_url)


def _evidence_to_prompt(evidence: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for item in evidence[:8]:
        content = item["content"].strip().replace("\n", " ")
        if len(content) > 1400:
            content = content[:1400] + "..."
        rows.append(
            "- "
            f"source={item['source_path']} "
            f"location={_location_to_text(item.get('location', {}))} "
            f"score={item['score']:.4f} "
            f"retrieval={item.get('retrieval_source', '')} "
            f"text={content}"
        )
    return "\n".join(rows) if rows else "(no evidence)"


def _build_chat_prompts(
    query: str,
    evidence: list[dict[str, Any]],
    query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    memory_context: str | None = None,
) -> tuple[str, str]:
    plan = _coerce_query_plan(query, query_plan)
    plan_text = _query_plan_to_prompt(plan)
    evidence_text = _evidence_to_prompt(evidence)
    system_prompt = (
        "You are a retrieval-grounded assistant for a RAG system.\n"
        "Answer only from the provided evidence. Do not invent facts.\n"
        "You may use the provided memory context only to resolve pronouns, ellipsis, or long-task continuity.\n"
        "Do not treat memory context as evidence; only the evidence section can support factual claims.\n"
        "Treat hard_terms as exact scope constraints. Do not mix entities, files, dates, ids, or report periods that violate them.\n"
        "Treat soft_terms as relevance hints, not hard filters.\n"
        "For ranking, list, or table questions, reconstruct the answer from the evidence in order.\n"
        "If adjacent chunks or pages from the same source continue the same section or table, combine them before answering.\n"
        "If the evidence is insufficient for an exact answer, say that briefly instead of guessing.\n"
        "Do not mention the query plan explicitly in the final answer."
    )
    user_prompt = f"Question:\n{query}\n\nQuery plan:\n{plan_text}\n\n"
    if memory_context:
        user_prompt += f"Memory context:\n{memory_context}\n\n"
    user_prompt += f"Evidence:\n{evidence_text}\n\nReturn a concise Chinese answer grounded in the evidence."
    return system_prompt, user_prompt


def _build_rerank_prompts(
    query: str,
    prompt_items: list[dict[str, Any]],
    query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
) -> tuple[str, str]:
    plan = _coerce_query_plan(query, query_plan)
    system_prompt = (
        "You are a reranker for a RAG system.\n"
        "Return JSON only as an array of objects: {\"evidence_id\": string, \"score\": float}.\n"
        "Higher scores must go to evidence that satisfies the hard_terms exactly, stays within the same entity and time scope, and directly supports the requested relation or answer shape.\n"
        "Evidence that only matches generic topic words should receive lower scores.\n"
        "If multiple adjacent chunks from the same source are jointly needed to answer the question, rank them high together."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Query plan:\n{_query_plan_to_prompt(plan)}\n\n"
        f"Candidates:\n{json.dumps(prompt_items, ensure_ascii=False)}"
    )
    return system_prompt, user_prompt


def _augment_rerank_query(
    query: str,
    query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
) -> str:
    plan = _coerce_query_plan(query, query_plan)
    if plan is None or (not plan.hard_terms and not plan.soft_terms):
        return query
    lines = [f"Question: {query}"]
    if plan.hard_terms:
        lines.append("Hard constraints: " + ", ".join(plan.hard_terms))
    if plan.soft_terms:
        lines.append("Soft hints: " + ", ".join(plan.soft_terms[:6]))
    if plan.answer_shape:
        lines.append(f"Answer shape: {plan.answer_shape}")
    return "\n".join(lines)


def _query_plan_to_prompt(query_plan: QueryConstraintPlan | None) -> str:
    if query_plan is None:
        return "hard_terms=[]; soft_terms=[]; intent=lookup; answer_shape=unknown"
    return (
        f"hard_terms={query_plan.hard_terms}; "
        f"soft_terms={query_plan.soft_terms}; "
        f"intent={query_plan.intent}; "
        f"answer_shape={query_plan.answer_shape}"
    )


def _location_to_text(location: Any) -> str:
    if isinstance(location, dict):
        parts = [f"{key}={value}" for key, value in location.items()]
        return "{" + ", ".join(parts) + "}"
    return str(location)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _extract_json_array(content: str) -> list[dict[str, Any]]:
    content = content.strip()
    if not content:
        return []
    start = content.find("[")
    end = content.rfind("]")
    if start < 0 or end <= start:
        return []
    payload = content[start : end + 1]
    try:
        data = json.loads(payload)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception:
        return []
    return []


def _coerce_query_plan(
    query: str, query_plan: QueryConstraintPlan | dict[str, Any] | None
) -> QueryConstraintPlan | None:
    if query_plan is None:
        return None
    if isinstance(query_plan, QueryConstraintPlan):
        return query_plan
    return query_plan_from_dict({"raw_query": query, **query_plan})
