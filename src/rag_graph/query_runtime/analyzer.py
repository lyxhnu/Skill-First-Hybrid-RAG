from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import Settings
from ..types import QueryConstraintPlan
from ..utils.text import extract_keywords, normalize_text

_FILE_PATTERN = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+\.(?:pdf|txt|md|xlsx|json)", re.IGNORECASE)
_TIME_PATTERN = re.compile(
    "(?:20\\d{2}(?:[-/]\\d{1,2}(?:[-/]\\d{1,2})?|"
    "\\u5e74(?:q[1-4]|\\u7b2c?[\\u4e00\\u4e8c\\u4e09\\u56db1-4]\\u5b63\\u5ea6|"
    "\\u4e0a\\u534a\\u5e74|\\u4e0b\\u534a\\u5e74)?))",
    re.IGNORECASE,
)
_QUESTION_MARKERS = (
    "top ",
    "\u524d\u5341",
    "\u524d\u4e09",
    "\u524d\u4e94",
    "\u54ea\u4e9b",
    "\u54ea\u4e2a",
    "\u591a\u5c11",
    "\u662f\u8c01",
    "\u662f\u4ec0\u4e48",
    "\u60c5\u51b5",
    "\u539f\u56e0",
    "\u8d8b\u52bf",
    "\u5206\u6790",
    "\u5217\u51fa",
    "\u7edf\u8ba1",
    "\u4ecb\u7ecd",
    "\u5bf9\u6bd4",
    "\u4e3a\u4ec0\u4e48",
    "\u5982\u4f55",
    "\u600e\u4e48",
    "\u662f\u5426",
)
_POLITE_PREFIXES = (
    "\u8bf7\u5e2e\u6211",
    "\u5e2e\u6211",
    "\u5e2e\u5fd9",
    "\u8bf7\u95ee",
    "\u8bf7\u4f60",
    "\u5206\u6790\u4e00\u4e0b",
    "\u5206\u6790\u4e0b",
    "\u770b\u4e00\u4e0b",
    "\u770b\u4e0b",
    "\u67e5\u4e00\u4e0b",
    "\u67e5\u4e0b",
)
_GENERIC_TERMS = {
    "\u6570\u636e",
    "\u4fe1\u606f",
    "\u5185\u5bb9",
    "\u60c5\u51b5",
    "\u95ee\u9898",
    "\u539f\u56e0",
    "\u8d8b\u52bf",
    "\u5206\u6790",
    "\u4ecb\u7ecd",
    "\u7edf\u8ba1",
    "\u5217\u51fa",
    "\u62a5\u544a",
    "\u8d22\u62a5",
}


class QueryConstraintAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(self, query: str) -> QueryConstraintPlan:
        heuristic = self._heuristic_plan(query)
        model_plan = self._model_plan(query)
        if model_plan is None:
            return heuristic
        expanded_plan = self._expand_model_plan(query, model_plan)
        return self._blend_plans(heuristic, expanded_plan)

    def refine_for_files(
        self,
        query: str,
        query_plan: QueryConstraintPlan | dict[str, Any] | None,
        candidate_files: list[str],
    ) -> QueryConstraintPlan:
        plan = _coerce_query_plan(query, query_plan)
        if plan is None or not candidate_files:
            return plan or self._heuristic_plan(query)
        extra_terms = self._model_file_aware_soft_terms(query, plan, candidate_files)
        if not extra_terms:
            return plan
        merged = list(plan.soft_terms)
        for term in extra_terms:
            if term in plan.hard_terms or term in merged:
                continue
            merged.append(term)
        metadata = dict(plan.metadata)
        metadata["file_aware_soft_terms"] = True
        return QueryConstraintPlan(
            raw_query=plan.raw_query,
            hard_terms=list(plan.hard_terms),
            soft_terms=merged[:10],
            intent=plan.intent,
            answer_shape=plan.answer_shape,
            metadata=metadata,
        )

    def _heuristic_plan(self, query: str) -> QueryConstraintPlan:
        normalized = normalize_text(query)
        hard_terms: list[str] = []
        soft_terms: list[str] = []

        for match in _FILE_PATTERN.findall(query):
            self._add_unique(hard_terms, match)

        for match in _TIME_PATTERN.findall(normalized):
            self._add_unique(hard_terms, match)

        scope = self._leading_scope_phrase(normalized)
        if scope:
            self._add_unique(hard_terms, scope)

        for token in extract_keywords(query):
            if token in hard_terms or token in _GENERIC_TERMS:
                continue
            self._add_unique(soft_terms, token)

        return QueryConstraintPlan(
            raw_query=query,
            hard_terms=hard_terms[:4],
            soft_terms=soft_terms[:10],
            intent=self._infer_intent(normalized),
            answer_shape=self._infer_answer_shape(normalized),
            metadata={"source": "heuristic"},
        )

    def _model_plan(self, query: str) -> QueryConstraintPlan | None:
        provider = self.settings.chat_provider.lower().strip()
        if provider in {"openai", "openai-compatible", "bailian", "zhipu"}:
            return self._openai_compatible_plan(query, provider)
        if provider == "gemini":
            return self._gemini_plan(query)
        return None

    def _openai_compatible_plan(self, query: str, provider: str) -> QueryConstraintPlan | None:
        api_key = ""
        base_url = None
        model = ""
        if provider in {"openai", "openai-compatible"}:
            api_key = self.settings.openai_api_key or ""
            base_url = self.settings.openai_base_url
            model = self.settings.openai_chat_model
        elif provider == "bailian":
            api_key = self.settings.bailian_api_key or ""
            base_url = self.settings.bailian_base_url
            model = self.settings.bailian_chat_model
        elif provider == "zhipu":
            api_key = self.settings.zhipu_api_key or ""
            base_url = self.settings.zhipu_base_url
            model = self.settings.zhipu_chat_model
        if not api_key or not model:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _QUERY_PLAN_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{query}\n\n"
                            "Return JSON only. Do not add markdown fences or explanation."
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content or ""
            return _plan_from_payload(query, _extract_json_object(content), source=f"model:{provider}")
        except Exception:
            return None

    def _gemini_plan(self, query: str) -> QueryConstraintPlan | None:
        if not self.settings.gemini_api_key:
            return None
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(model_name=self.settings.gemini_chat_model)
            response = model.generate_content(
                f"{_QUERY_PLAN_SYSTEM_PROMPT}\n\nQuestion:\n{query}\n\nReturn JSON only."
            )
            return _plan_from_payload(query, _extract_json_object(response.text or ""), source="model:gemini")
        except Exception:
            return None

    def _expand_model_plan(self, query: str, plan: QueryConstraintPlan) -> QueryConstraintPlan:
        if len(plan.soft_terms) >= 2:
            return plan
        extra_terms = self._model_soft_term_expansion(query, plan)
        if not extra_terms:
            return plan
        merged = list(plan.soft_terms)
        for term in extra_terms:
            if term in plan.hard_terms or term in merged:
                continue
            merged.append(term)
        metadata = dict(plan.metadata)
        metadata["soft_term_expanded"] = True
        return QueryConstraintPlan(
            raw_query=plan.raw_query,
            hard_terms=list(plan.hard_terms),
            soft_terms=merged[:10],
            intent=plan.intent,
            answer_shape=plan.answer_shape,
            metadata=metadata,
        )

    def _model_soft_term_expansion(self, query: str, plan: QueryConstraintPlan) -> list[str]:
        provider = self.settings.chat_provider.lower().strip()
        if provider in {"openai", "openai-compatible", "bailian", "zhipu"}:
            return self._openai_compatible_soft_terms(query, plan, provider)
        if provider == "gemini":
            return self._gemini_soft_terms(query, plan)
        return []

    def _model_file_aware_soft_terms(
        self,
        query: str,
        plan: QueryConstraintPlan,
        candidate_files: list[str],
    ) -> list[str]:
        provider = self.settings.chat_provider.lower().strip()
        if provider in {"openai", "openai-compatible", "bailian", "zhipu"}:
            return self._openai_compatible_file_aware_soft_terms(query, plan, candidate_files, provider)
        if provider == "gemini":
            return self._gemini_file_aware_soft_terms(query, plan, candidate_files)
        return []

    def _openai_compatible_soft_terms(
        self, query: str, plan: QueryConstraintPlan, provider: str
    ) -> list[str]:
        api_key = ""
        base_url = None
        model = ""
        if provider in {"openai", "openai-compatible"}:
            api_key = self.settings.openai_api_key or ""
            base_url = self.settings.openai_base_url
            model = self.settings.openai_chat_model
        elif provider == "bailian":
            api_key = self.settings.bailian_api_key or ""
            base_url = self.settings.bailian_base_url
            model = self.settings.bailian_chat_model
        elif provider == "zhipu":
            api_key = self.settings.zhipu_api_key or ""
            base_url = self.settings.zhipu_base_url
            model = self.settings.zhipu_chat_model
        if not api_key or not model:
            return []

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _SOFT_TERM_EXPANSION_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{query}\n\n"
                            f"hard_terms={plan.hard_terms}\n"
                            f"current_soft_terms={plan.soft_terms}\n\n"
                            "Return JSON only."
                        ),
                    },
                ],
            )
            payload = _extract_json_object(response.choices[0].message.content or "")
            return _sanitize_extra_soft_terms(payload, query, plan)
        except Exception:
            return []

    def _openai_compatible_file_aware_soft_terms(
        self,
        query: str,
        plan: QueryConstraintPlan,
        candidate_files: list[str],
        provider: str,
    ) -> list[str]:
        api_key = ""
        base_url = None
        model = ""
        if provider in {"openai", "openai-compatible"}:
            api_key = self.settings.openai_api_key or ""
            base_url = self.settings.openai_base_url
            model = self.settings.openai_chat_model
        elif provider == "bailian":
            api_key = self.settings.bailian_api_key or ""
            base_url = self.settings.bailian_base_url
            model = self.settings.bailian_chat_model
        elif provider == "zhipu":
            api_key = self.settings.zhipu_api_key or ""
            base_url = self.settings.zhipu_base_url
            model = self.settings.zhipu_chat_model
        if not api_key or not model:
            return []

        file_names = [Path(path).name for path in candidate_files[:6]]
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _FILE_AWARE_SOFT_TERM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{query}\n\n"
                            f"hard_terms={plan.hard_terms}\n"
                            f"current_soft_terms={plan.soft_terms}\n"
                            f"candidate_files={file_names}\n\n"
                            "Return JSON only."
                        ),
                    },
                ],
            )
            payload = _extract_json_object(response.choices[0].message.content or "")
            return _sanitize_extra_soft_terms(payload, query, plan)
        except Exception:
            return []

    def _gemini_soft_terms(self, query: str, plan: QueryConstraintPlan) -> list[str]:
        if not self.settings.gemini_api_key:
            return []
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(model_name=self.settings.gemini_chat_model)
            response = model.generate_content(
                f"{_SOFT_TERM_EXPANSION_PROMPT}\n\n"
                f"Question:\n{query}\n\n"
                f"hard_terms={plan.hard_terms}\n"
                f"current_soft_terms={plan.soft_terms}\n\n"
                "Return JSON only."
            )
            payload = _extract_json_object(response.text or "")
            return _sanitize_extra_soft_terms(payload, query, plan)
        except Exception:
            return []

    def _gemini_file_aware_soft_terms(
        self,
        query: str,
        plan: QueryConstraintPlan,
        candidate_files: list[str],
    ) -> list[str]:
        if not self.settings.gemini_api_key:
            return []
        file_names = [Path(path).name for path in candidate_files[:6]]
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(model_name=self.settings.gemini_chat_model)
            response = model.generate_content(
                f"{_FILE_AWARE_SOFT_TERM_PROMPT}\n\n"
                f"Question:\n{query}\n\n"
                f"hard_terms={plan.hard_terms}\n"
                f"current_soft_terms={plan.soft_terms}\n"
                f"candidate_files={file_names}\n\n"
                "Return JSON only."
            )
            payload = _extract_json_object(response.text or "")
            return _sanitize_extra_soft_terms(payload, query, plan)
        except Exception:
            return []

    def _blend_plans(self, heuristic: QueryConstraintPlan, model_plan: QueryConstraintPlan) -> QueryConstraintPlan:
        hard_terms = list(model_plan.hard_terms or heuristic.hard_terms)
        soft_terms = list(model_plan.soft_terms or heuristic.soft_terms)
        return QueryConstraintPlan(
            raw_query=heuristic.raw_query,
            hard_terms=hard_terms[:4],
            soft_terms=soft_terms[:10],
            intent=model_plan.intent or heuristic.intent,
            answer_shape=model_plan.answer_shape or heuristic.answer_shape,
            metadata={
                "source": model_plan.metadata.get("source", "model"),
                "fallback_used": {
                    "hard_terms": not bool(model_plan.hard_terms),
                    "soft_terms": not bool(model_plan.soft_terms),
                },
                "soft_term_expanded": bool(model_plan.metadata.get("soft_term_expanded")),
            },
        )

    @staticmethod
    def _leading_scope_phrase(query: str) -> str:
        scope = query.strip(" ,.!?;:，。！？；：")
        for prefix in _POLITE_PREFIXES:
            if scope.startswith(prefix):
                scope = scope[len(prefix) :].strip()
        cut_positions = [scope.find(marker) for marker in _QUESTION_MARKERS if scope.find(marker) > 0]
        if cut_positions:
            scope = scope[: min(cut_positions)]
        scope = scope.strip(" ,.!?;:，。！？；：")
        if len(scope) < 2 or scope in _GENERIC_TERMS:
            return ""
        return scope

    @staticmethod
    def _infer_intent(normalized: str) -> str:
        if any(marker in normalized for marker in ("\u5206\u6790", "\u539f\u56e0", "\u8d8b\u52bf", "\u5bf9\u6bd4", "\u6bd4\u8f83")):
            return "analysis"
        if any(marker in normalized for marker in ("\u591a\u5c11", "\u6570\u91cf", "\u51e0\u4e2a", "\u51e0\u5bb6")):
            return "count"
        if any(marker in normalized for marker in ("\u5217\u51fa", "\u540d\u5355", "\u524d\u5341", "\u524d\u4e09", "top")):
            return "list_lookup"
        return "lookup"

    @staticmethod
    def _infer_answer_shape(normalized: str) -> str:
        if any(marker in normalized for marker in ("\u524d\u5341", "\u524d\u4e09", "\u524d\u4e94", "\u540d\u5355", "\u5217\u51fa", "top")):
            return "list"
        if any(marker in normalized for marker in ("\u591a\u5c11", "\u6570\u91cf", "\u51e0\u4e2a", "\u51e0\u5bb6")):
            return "count"
        if any(marker in normalized for marker in ("\u5bf9\u6bd4", "\u6bd4\u8f83")):
            return "comparison"
        if "\u8868" in normalized:
            return "table"
        return "fact"

    @staticmethod
    def _add_unique(target: list[str], term: str) -> None:
        normalized = normalize_text(term)
        if not normalized or normalized in target:
            return
        target.append(normalized)


_QUERY_PLAN_SYSTEM_PROMPT = """
You are a query planner for a RAG system.
Return JSON only with this schema:
{"hard_terms":["..."],"soft_terms":["..."],"intent":"...","answer_shape":"..."}

Definitions:
- hard_terms: exact scope constraints that should stay consistent across retrieved evidence.
  Use them for entity names, person names, product names, filenames, ids, report periods, dates, regions, or other exact scopes.
- soft_terms: topic, relation, section, metric, or operation hints.
  Ranking phrases, relationship phrases, analysis goals, and attributes belong here.
  soft_terms may also include likely section names, table titles, or source phrases that could appear verbatim in documents.

Rules:
- Prefer a small number of precise hard_terms.
- Do not put generic question words into hard_terms.
- Do not put relation phrases such as ranking/comparison/filter instructions into hard_terms unless they are literal file names or ids.
- If the user asks for "A's top 3 shareholders", "A" is a hard_term and "top 3 shareholders" is a soft_term.
- If the user asks to analyze a table or spreadsheet without naming an exact entity, hard_terms may be empty.
- Do not repeat the whole question in soft_terms.
- Do not generate broken substrings, character n-grams, or near-duplicate variants.
- Do not copy hard_terms into soft_terms unless the hard term itself is also the literal target field.
- When the user asks for ranked items, table rows, or report facts, include the likely section title or table title that may contain the answer.
- When the source is likely a JSON/FAQ knowledge base, include likely service topics, policy names, or FAQ-style question phrases in soft_terms.
- Prefer 1-4 high-value soft_terms that improve retrieval, not many variants of the same phrase.
- intent should be a short snake_case label.
- answer_shape should be one of: fact, list, table, count, comparison, summary, unknown.

Examples:
Question: SANY top 3 shareholders
{"hard_terms":["SANY"],"soft_terms":["top 3 shareholders","top 10 shareholders","shareholder holdings"],"intent":"entity_lookup","answer_shape":"list"}

Question: \u4e09\u4e00\u91cd\u5de5\u524d\u4e09\u5927\u80a1\u4e1c
{"hard_terms":["\u4e09\u4e00\u91cd\u5de5"],"soft_terms":["\u524d\u4e09\u5927\u80a1\u4e1c","\u524d10\u540d\u80a1\u4e1c","\u80a1\u4e1c\u6301\u80a1\u60c5\u51b5"],"intent":"entity_lookup","answer_shape":"list"}

Question: analyze inventory data and find low-stock products
{"hard_terms":[],"soft_terms":["low stock","reorder level","inventory sheet"],"intent":"analysis","answer_shape":"list"}

Question: \u5206\u6790\u5e93\u5b58\u6570\u636e\uff0c\u54ea\u4e9b\u5546\u54c1\u5e93\u5b58\u4e0d\u8db3
{"hard_terms":[],"soft_terms":["\u5e93\u5b58\u4e0d\u8db3","\u8865\u8d27\u9608\u503c","\u5e93\u5b58\u8868"],"intent":"analysis","answer_shape":"list"}

Question: \u8ba2\u5355\u53d6\u6d88\u653f\u7b56\u662f\u4ec0\u4e48
{"hard_terms":[],"soft_terms":["\u53d6\u6d88\u653f\u7b56","\u8ba2\u5355\u53d6\u6d88","\u9000\u6b3e\u89c4\u5219","\u5e38\u89c1\u95ee\u9898"],"intent":"policy_lookup","answer_shape":"fact"}
""".strip()

_SOFT_TERM_EXPANSION_PROMPT = """
You expand retrieval-friendly soft terms for a RAG system.
Return JSON only with this schema:
{"soft_terms":["..."]}

Rules:
- Keep hard_terms unchanged. Do not repeat them.
- Add 1-4 extra phrases that are likely to appear verbatim in documents, such as section titles, table titles, metric names, or equivalent source phrases.
- Do not repeat the whole question.
- Do not output broken substrings, n-grams, or near-duplicate variants.
- Prefer phrases that improve retrieval in reports, PDFs, spreadsheets, and text chunks.

Examples:
Question: SANY top 3 shareholders
hard_terms=["SANY"]
current_soft_terms=["top 3 shareholders"]
{"soft_terms":["top 10 shareholders","shareholder holdings"]}

Question: \u4e09\u4e00\u91cd\u5de5\u524d\u4e09\u5927\u80a1\u4e1c
hard_terms=["\u4e09\u4e00\u91cd\u5de5"]
current_soft_terms=["\u524d\u4e09\u5927\u80a1\u4e1c"]
{"soft_terms":["\u524d10\u540d\u80a1\u4e1c","\u80a1\u4e1c\u6301\u80a1\u60c5\u51b5"]}

Question: low inventory products
hard_terms=[]
current_soft_terms=["low stock"]
{"soft_terms":["reorder level","inventory sheet"]}

Question: \u5e93\u5b58\u4e0d\u8db3\u7684\u5546\u54c1
hard_terms=[]
current_soft_terms=["\u5e93\u5b58\u4e0d\u8db3"]
{"soft_terms":["\u8865\u8d27\u9608\u503c","\u5e93\u5b58\u8868"]}

Question: \u8ba2\u5355\u53d6\u6d88\u653f\u7b56\u662f\u4ec0\u4e48
hard_terms=[]
current_soft_terms=["\u53d6\u6d88\u653f\u7b56"]
{"soft_terms":["\u8ba2\u5355\u53d6\u6d88","\u9000\u6b3e\u89c4\u5219","\u5e38\u89c1\u95ee\u9898"]}
""".strip()

_FILE_AWARE_SOFT_TERM_PROMPT = """
You refine retrieval-friendly soft terms for a RAG system after candidate files have been selected.
Return JSON only with this schema:
{"soft_terms":["..."]}

Rules:
- Keep hard_terms unchanged. Do not repeat them.
- Use candidate file names as weak context only. Infer the most likely section titles, table titles, metric names, or source phrases that may appear inside those files.
- Prefer broader document headings over paraphrases of the user question.
- Do not output broken substrings, n-grams, or near-duplicate variants.
- Prefer 1-4 additions that improve recall inside the selected files.

Examples:
Question: \u4e09\u4e00\u91cd\u5de5\u524d\u4e09\u5927\u80a1\u4e1c
hard_terms=["\u4e09\u4e00\u91cd\u5de5"]
current_soft_terms=["\u524d\u4e09\u5927\u80a1\u4e1c"]
candidate_files=["\u4e09\u4e00\u91cd\u5de5 2025 Q3.pdf","\u4e09\u4e00\u91cd\u5de5_2025_Q3.txt"]
{"soft_terms":["\u524d10\u540d\u80a1\u4e1c","\u80a1\u4e1c\u6301\u80a1\u60c5\u51b5"]}

Question: \u5982\u4f55\u53d6\u6d88\u8ba2\u5355
hard_terms=[]
current_soft_terms=["\u53d6\u6d88\u8ba2\u5355"]
candidate_files=["faq.json"]
{"soft_terms":["\u53d6\u6d88\u653f\u7b56","\u9000\u6b3e\u89c4\u5219","\u5e38\u89c1\u95ee\u9898"]}
""".strip()


def _plan_from_payload(query: str, payload: dict[str, Any], source: str) -> QueryConstraintPlan | None:
    if not payload:
        return None
    hard_terms = _sanitize_terms(payload.get("hard_terms", []))[:4]
    raw_query = normalize_text(query)
    soft_terms = [
        term
        for term in _sanitize_terms(payload.get("soft_terms", []))
        if term != raw_query and term not in hard_terms
    ][:10]
    return QueryConstraintPlan(
        raw_query=query,
        hard_terms=hard_terms,
        soft_terms=soft_terms,
        intent=_sanitize_scalar(payload.get("intent"), default="lookup"),
        answer_shape=_sanitize_scalar(payload.get("answer_shape"), default="unknown"),
        metadata={"source": source},
    )


def _sanitize_extra_soft_terms(
    payload: dict[str, Any],
    query: str,
    plan: QueryConstraintPlan,
) -> list[str]:
    raw_query = normalize_text(query)
    terms: list[str] = []
    for term in _sanitize_terms(payload.get("soft_terms", [])):
        if term == raw_query or term in plan.hard_terms or term in plan.soft_terms:
            continue
        terms.append(term)
    return terms[:4]


def _extract_json_object(content: str) -> dict[str, Any]:
    content = (content or "").strip()
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(content[start : end + 1])
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sanitize_terms(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    terms: list[str] = []
    for value in values:
        normalized = normalize_text(str(value))
        if not normalized or normalized in terms:
            continue
        terms.append(normalized)
    return terms


def _sanitize_scalar(value: Any, default: str) -> str:
    normalized = normalize_text(str(value or ""))
    return normalized or default


def _coerce_query_plan(
    query: str,
    query_plan: QueryConstraintPlan | dict[str, Any] | None,
) -> QueryConstraintPlan | None:
    if query_plan is None:
        return None
    if isinstance(query_plan, QueryConstraintPlan):
        return query_plan
    payload = dict(query_plan)
    payload.setdefault("raw_query", query)
    return QueryConstraintPlan(
        raw_query=str(payload.get("raw_query", "")),
        hard_terms=[str(item) for item in payload.get("hard_terms", []) if str(item).strip()],
        soft_terms=[str(item) for item in payload.get("soft_terms", []) if str(item).strip()],
        intent=str(payload.get("intent", "lookup") or "lookup"),
        answer_shape=str(payload.get("answer_shape", "unknown") or "unknown"),
        metadata=dict(payload.get("metadata", {})),
    )
