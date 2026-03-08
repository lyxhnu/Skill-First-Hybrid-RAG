from __future__ import annotations

import json
from typing import Any

from ..config import Settings
from ..types import QueryConstraintPlan
from ..utils.text import lexical_score, normalize_text
from .store import MemoryStore


ALLOWED_LONG_TERM_CATEGORIES = {"project", "preference", "constraint", "goal", "identity"}
PROJECT_MEMORY_HINTS = ("项目", "系统", "工作区", "仓库", "服务", "流程", "架构", "pipeline", "workflow", "langgraph", "rag", "skill")


class MemoryManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = MemoryStore(settings)

    def build_context(self, *, session_id: str, actor_id: str, query: str) -> dict[str, Any]:
        self.store.ensure_session(session_id, actor_id)
        turns = self.store.load_turns(session_id)
        summaries = self.store.load_summaries(session_id)
        long_term = self.store.load_long_term(actor_id)

        window_turns = turns[-self.settings.memory_window_turns :]
        summary_blocks = self._rank_summary_blocks(query, summaries)
        recalled_turns = self._recall_turns_from_summaries(query, summary_blocks)
        long_term_hits = self._rank_long_term_memories(query, long_term)

        prompt_text = self._format_memory_context(
            long_term_hits=long_term_hits,
            summary_blocks=summary_blocks,
            recalled_turns=recalled_turns,
            window_turns=window_turns,
        )
        rewrite_context = self._format_rewrite_context(
            summary_blocks=summary_blocks,
            recalled_turns=recalled_turns,
            window_turns=window_turns,
        )
        effective_query = query
        contextualized = False
        if rewrite_context:
            rewritten = self._rewrite_query(query, rewrite_context)
            if rewritten:
                effective_query = rewritten
                contextualized = normalize_text(rewritten) != normalize_text(query)

        return {
            "session_id": session_id,
            "actor_id": actor_id,
            "effective_query": effective_query,
            "prompt": prompt_text,
            "trace": {
                "session_id": session_id,
                "actor_id": actor_id,
                "window_turn_ids": [int(row.get("turn_id", 0)) for row in window_turns],
                "summary_block_ids": [str(row.get("summary_id", "")) for row in summary_blocks],
                "recalled_turn_ids": [int(row.get("turn_id", 0)) for row in recalled_turns],
                "long_term_ids": [str(row.get("memory_id", "")) for row in long_term_hits],
                "contextualized": contextualized,
                "effective_query": effective_query,
            },
        }

    def persist_turn(
        self,
        *,
        session_id: str,
        actor_id: str,
        user_query: str,
        effective_query: str,
        answer: str,
        citations: list[dict[str, Any]],
        query_constraints: dict[str, Any],
    ) -> dict[str, Any]:
        turn = self.store.append_turn(
            session_id=session_id,
            actor_id=actor_id,
            user_query=user_query,
            effective_query=effective_query,
            answer=answer,
            citations=citations,
            query_constraints=query_constraints,
        )
        summary_ids = self._summarize_if_needed(session_id=session_id, actor_id=actor_id)
        return {"turn_id": turn["turn_id"], "summary_ids": summary_ids}

    def session_stats(self, session_id: str, actor_id: str) -> dict[str, Any]:
        meta = self.store.ensure_session(session_id, actor_id)
        return {
            "session_id": session_id,
            "actor_id": actor_id,
            "turn_count": int(meta.get("turn_count", 0)),
            "summary_count": int(meta.get("summary_count", 0)),
            "last_summarized_turn_id": int(meta.get("last_summarized_turn_id", 0)),
        }

    def _summarize_if_needed(self, *, session_id: str, actor_id: str) -> list[str]:
        created: list[str] = []
        meta = self.store.ensure_session(session_id, actor_id)
        turns = self.store.load_turns(session_id)
        start_turn = int(meta.get("last_summarized_turn_id", 0)) + 1
        unsummarized = [row for row in turns if int(row.get("turn_id", 0)) >= start_turn]
        block_size = max(2, int(self.settings.memory_summary_block_turns))
        trigger = max(block_size, int(self.settings.memory_summary_trigger_turns))

        while len(unsummarized) >= trigger:
            block_turns = unsummarized[:block_size]
            payload = self._summarize_turn_block(session_id, actor_id, block_turns)
            summary = self.store.append_summary(session_id, payload)
            created.append(str(summary.get("summary_id", "")))
            for candidate in payload.get("long_term_candidates", []):
                self._append_long_term_candidate(actor_id, session_id, candidate, summary)
            meta = self.store.load_session_meta(session_id)
            start_turn = int(meta.get("last_summarized_turn_id", 0)) + 1
            unsummarized = [row for row in turns if int(row.get("turn_id", 0)) >= start_turn]
        return created

    def _summarize_turn_block(
        self,
        session_id: str,
        actor_id: str,
        turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = self._model_summary_payload(session_id, actor_id, turns)
        if not payload:
            payload = self._heuristic_summary_payload(turns)
        turn_ids = [int(row.get("turn_id", 0)) for row in turns]
        source_paths = [self.store.turn_path(session_id, turn_id) for turn_id in turn_ids]
        return {
            "session_id": session_id,
            "actor_id": actor_id,
            "turn_ids": turn_ids,
            "source_paths": source_paths,
            "summary": str(payload.get("summary", "")).strip(),
            "key_points": [str(item).strip() for item in payload.get("key_points", []) if str(item).strip()][:8],
            "open_questions": [str(item).strip() for item in payload.get("open_questions", []) if str(item).strip()][:5],
            "long_term_candidates": payload.get("long_term_candidates", []),
        }

    def _append_long_term_candidate(
        self,
        actor_id: str,
        session_id: str,
        candidate: dict[str, Any],
        summary: dict[str, Any],
    ) -> None:
        if not isinstance(candidate, dict):
            return
        text = str(candidate.get("text", "")).strip()
        if len(text) < 4:
            return
        category = str(candidate.get("category", "")).strip().lower()
        if category not in ALLOWED_LONG_TERM_CATEGORIES:
            return
        importance = max(0.0, min(1.0, float(candidate.get("importance", 0.0) or 0.0)))
        if importance < 0.6:
            return
        if category == "project" and not any(hint in text.lower() for hint in PROJECT_MEMORY_HINTS):
            return
        normalized = normalize_text(text)
        existing = self.store.load_long_term(actor_id)
        if normalized in {normalize_text(str(row.get("text", ""))) for row in existing}:
            return
        self.store.append_long_term(
            actor_id,
            {
                "actor_id": actor_id,
                "session_id": session_id,
                "text": text,
                "category": category or "project",
                "importance": importance,
                "source_paths": list(summary.get("source_paths", [])),
                "summary_id": summary.get("summary_id", ""),
            },
        )

    def _rank_summary_blocks(self, query: str, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not summaries:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, summary in enumerate(summaries):
            payload = "\n".join(
                [
                    str(summary.get("summary", "")),
                    " ".join(str(item) for item in summary.get("key_points", [])),
                    " ".join(str(item) for item in summary.get("open_questions", [])),
                ]
            )
            score = lexical_score(query, payload) + idx * 0.03
            scored.append((score, summary))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [row for score, row in scored if score > 0][: self.settings.memory_summary_top_k]
        if selected:
            return selected
        return summaries[-1:] if summaries else []

    def _recall_turns_from_summaries(self, query: str, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not summaries:
            return []
        paths: list[str] = []
        for summary in summaries:
            paths.extend(str(path) for path in summary.get("source_paths", []))
        turns = self.store.resolve_turn_paths(paths)
        if not turns:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, turn in enumerate(turns):
            payload = f"{turn.get('user_query', '')}\n{turn.get('answer', '')}"
            score = lexical_score(query, payload) + idx * 0.01
            scored.append((score, turn))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [row for score, row in scored if score > 0][: self.settings.memory_recall_turns]
        if selected:
            return sorted(selected, key=lambda row: int(row.get("turn_id", 0)))
        return turns[-self.settings.memory_recall_turns :]

    def _rank_long_term_memories(self, query: str, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not memories:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, memory in enumerate(memories):
            payload = f"{memory.get('text', '')}\n{memory.get('category', '')}"
            score = lexical_score(query, payload) + float(memory.get("importance", 0.0)) + idx * 0.01
            scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [row for score, row in scored if score > 0][: self.settings.long_term_memory_top_k]
        if selected:
            return selected
        return memories[-1:] if memories else []

    @staticmethod
    def _format_memory_context(
        *,
        long_term_hits: list[dict[str, Any]],
        summary_blocks: list[dict[str, Any]],
        recalled_turns: list[dict[str, Any]],
        window_turns: list[dict[str, Any]],
    ) -> str:
        sections: list[str] = []
        if long_term_hits:
            rows = [f"- {row.get('text', '')}" for row in long_term_hits]
            sections.append("[LongTermMemory]\n" + "\n".join(rows))
        if summary_blocks:
            rows = []
            for summary in summary_blocks:
                key_points = "；".join(str(item) for item in summary.get("key_points", [])[:4])
                rows.append(f"- {summary.get('summary', '')}" + (f" | key_points={key_points}" if key_points else ""))
            sections.append("[SummaryBlocks]\n" + "\n".join(rows))
        if recalled_turns:
            rows = []
            for turn in recalled_turns:
                rows.append(
                    f"- turn#{turn.get('turn_id')}: user={turn.get('user_query', '')} | assistant={turn.get('answer', '')}"
                )
            sections.append("[RecalledTurns]\n" + "\n".join(rows))
        if window_turns:
            rows = []
            for turn in window_turns:
                rows.append(
                    f"- turn#{turn.get('turn_id')}: user={turn.get('user_query', '')} | assistant={turn.get('answer', '')}"
                )
            sections.append("[SlidingWindow]\n" + "\n".join(rows))
        return "\n\n".join(section for section in sections if section.strip())

    @staticmethod
    def _format_rewrite_context(
        *,
        summary_blocks: list[dict[str, Any]],
        recalled_turns: list[dict[str, Any]],
        window_turns: list[dict[str, Any]],
    ) -> str:
        sections: list[str] = []
        if window_turns:
            latest_turn = window_turns[-1]
            sections.append(
                "[MostRecentTurn]\n"
                + f"- turn#{latest_turn.get('turn_id')}: user={latest_turn.get('user_query', '')} | assistant={latest_turn.get('answer', '')}"
            )
            earlier_turns = window_turns[:-1][-3:]
            if earlier_turns:
                rows = []
                for turn in reversed(earlier_turns):
                    rows.append(
                        f"- turn#{turn.get('turn_id')}: user={turn.get('user_query', '')} | assistant={turn.get('answer', '')}"
                    )
                sections.append("[EarlierRecentTurns]\n" + "\n".join(rows))
        if recalled_turns:
            rows = []
            for turn in reversed(recalled_turns[-3:]):
                rows.append(
                    f"- turn#{turn.get('turn_id')}: user={turn.get('user_query', '')} | assistant={turn.get('answer', '')}"
                )
            sections.append("[RecalledTurns]\n" + "\n".join(rows))
        if summary_blocks:
            rows = [f"- {row.get('summary', '')}" for row in summary_blocks[:1]]
            sections.append("[RelevantSummary]\n" + "\n".join(rows))
        return "\n\n".join(section for section in sections if section.strip())

    def _rewrite_query(self, query: str, memory_context: str) -> str:
        provider = self.settings.chat_provider.lower().strip()
        reference = self._select_reference_turn(query, memory_context)
        resolved_subject = str(reference.get("resolved_subject", "")).strip()
        target_turn_id = int(reference.get("target_turn_id", 0) or 0)
        if target_turn_id <= 0:
            return query
        system_prompt = (
            "You rewrite follow-up user questions into standalone questions for retrieval.\n"
            "Return JSON only with this schema:\n"
            "{\"standalone_question\":\"...\",\"resolved_subject\":\"...\",\"reason\":\"...\"}\n"
            "Rules:\n"
            "- Rewrite only when the selected antecedent turn provides essential missing context.\n"
            "- Carry forward only the context needed to make the current question standalone.\n"
            "- Missing context may include entity, person, company, product, document, sheet, file, table, time range, list/ranking scope, filters, metric, comparison baseline, or the target being explained.\n"
            "- Prefer the selected antecedent turn first, then use nearby turns only when they add necessary constraints.\n"
            "- Preserve the user's current intent. Do not answer the question; only rewrite it.\n"
            "- If the current question modifies one dimension of the prior topic, keep the unchanged constraints and replace only the changed part.\n"
            "- If the current question asks for source, reason, definition, method, next item, comparison, or narrowed scope, keep the same topic target from context when needed.\n"
            "- Resolve pronouns, deictic references, omitted subjects, omitted filters, and omitted metrics to explicit text whenever the context makes them clear.\n"
            "- Keep distinguishing attributes when required to avoid ambiguity, such as rank, highest/lowest, time period, department, region, sheet, or document name.\n"
            "- Do not drag irrelevant context into a new standalone topic.\n"
            "- If the selected antecedent still does not supply enough missing context, return the original question unchanged.\n"
            "- reason must be short.\n"
            "Examples:\n"
            "- Previous answer: 罗凯的工资最高，为34900元。 Follow-up: 他在哪个部门 -> {\"standalone_question\":\"工资最高的员工罗凯在哪个部门工作？\",\"resolved_subject\":\"工资最高的员工罗凯\",\"reason\":\"fill omitted person from antecedent\"}\n"
            "- Previous answer: 销售部门的职工有刘洋、韩磊、林月、马洁、许晨。 Follow-up: 那上海的呢 -> {\"standalone_question\":\"销售部门里上海的职工有哪些？\",\"resolved_subject\":\"销售部门职工\",\"reason\":\"keep same scope and add city filter\"}\n"
            "- Previous answer: 2025 Q3 三一重工的营收为... Follow-up: 那净利润呢 -> {\"standalone_question\":\"2025 Q3 三一重工的净利润是多少？\",\"resolved_subject\":\"2025 Q3 三一重工财务指标\",\"reason\":\"same entity and time range, different metric\"}\n"
            "- Previous answer: inventory.xlsx 中库存不足的商品有儿童绘本、移动硬盘、路由器。 Follow-up: 给我前三个 -> {\"standalone_question\":\"inventory.xlsx 中库存不足的商品里，前三个是什么？\",\"resolved_subject\":\"inventory.xlsx 库存不足商品列表\",\"reason\":\"continue current list scope\"}"
        )
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Memory context:\n{memory_context}\n\n"
            f"Selected antecedent turn id: {target_turn_id}\n"
            f"Resolved subject hint: {resolved_subject or '(none)'}\n\n"
            "Return JSON only."
        )

        payload: dict[str, Any] = {}
        if provider in {"openai", "openai-compatible", "bailian", "zhipu"}:
            payload = self._openai_compatible_json(provider, system_prompt, user_prompt)
        elif provider == "gemini":
            payload = self._gemini_json(system_prompt, user_prompt)
        if payload:
            rewritten = str(payload.get("standalone_question", "")).strip()
            if rewritten:
                return rewritten
        return query

    def _select_reference_turn(self, query: str, memory_context: str) -> dict[str, Any]:
        provider = self.settings.chat_provider.lower().strip()
        system_prompt = (
            "You identify which prior turn a follow-up question refers to.\n"
            "Return JSON only with this schema:\n"
            "{\"target_turn_id\":0,\"resolved_subject\":\"...\",\"reason\":\"...\"}\n"
            "Rules:\n"
            "- Select 0 when the current question is already self-contained and understandable without prior turns.\n"
            "- Do not attach an antecedent just because a session has previous turns.\n"
            "- Do not rewrite or attach context for a new standalone topic.\n"
            "- Prefer the most recent relevant turn over older topics.\n"
            "- Use [MostRecentTurn] first, but switch to [EarlierRecentTurns] or [RecalledTurns] when the latest turn is unrelated or only partially relevant.\n"
            "- Choose the prior turn that supplies the missing context necessary to understand the current question.\n"
            "- Missing context may include omitted subject, omitted entity, omitted time range, omitted document/sheet/table, omitted filters, omitted metric, omitted ranking/list scope, omitted comparison baseline, or omitted target of explanation.\n"
            "- Follow-up questions can be phrased as pronouns, deictic expressions, ellipsis, narrowed scope, changed metric, changed filter, request for explanation, request for source, request for comparison, request for next item, or shortened restatements.\n"
            "- If the current question continues the same topic but changes one dimension, link it to the turn that best defines the unchanged context.\n"
            "- When the relevant prior answer contains a distinguishing attribute for a named subject, preserve that attribute in resolved_subject reasoning.\n"
            "- If no antecedent is needed because the question is self-contained, return 0 and an empty subject.\n"
            "Examples:\n"
            "- Latest answer: 罗凯的工资最高，为34900元。 Follow-up: 他在哪个部门 -> {\"target_turn_id\":5,\"resolved_subject\":\"工资最高的员工罗凯\",\"reason\":\"antecedent supplies the omitted person\"}\n"
            "- Previous answer: 销售部门的职工有刘洋、韩磊、林月、马洁、许晨。 Follow-up: 那上海的呢 -> {\"target_turn_id\":3,\"resolved_subject\":\"销售部门职工列表\",\"reason\":\"antecedent supplies the scope while current question adds a city filter\"}\n"
            "- Previous answer: 2025 Q3 三一重工的营收为... Follow-up: 那净利润呢 -> {\"target_turn_id\":4,\"resolved_subject\":\"2025 Q3 三一重工财务指标\",\"reason\":\"antecedent supplies entity and time range\"}\n"
            "- Previous turn is about shareholders, current question is 哪些商品库存不足 -> {\"target_turn_id\":0,\"resolved_subject\":\"\",\"reason\":\"current question is a new standalone topic\"}"
        )
        user_prompt = f"Question:\n{query}\n\nMemory context:\n{memory_context}\n\nReturn JSON only."

        payload: dict[str, Any] = {}
        if provider in {"openai", "openai-compatible", "bailian", "zhipu"}:
            payload = self._openai_compatible_json(provider, system_prompt, user_prompt)
        elif provider == "gemini":
            payload = self._gemini_json(system_prompt, user_prompt)
        if not payload:
            return {}
        turn_id = int(payload.get("target_turn_id", 0) or 0)
        return {
            "target_turn_id": max(0, turn_id),
            "resolved_subject": str(payload.get("resolved_subject", "")).strip(),
            "reason": str(payload.get("reason", "")).strip(),
        }

    def _model_summary_payload(
        self,
        session_id: str,
        actor_id: str,
        turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        provider = self.settings.chat_provider.lower().strip()
        system_prompt = (
            "You summarize conversation turns into compact memory blocks for a RAG system.\n"
            "Return JSON only with this schema:\n"
            "{\"summary\":\"...\",\"key_points\":[\"...\"],\"open_questions\":[\"...\"],"
            "\"long_term_candidates\":[{\"text\":\"...\",\"category\":\"project\",\"importance\":0.0}]}\n"
            "Rules:\n"
            "- Keep summary short but precise.\n"
            "- key_points should preserve facts needed for future retrieval inside the same session.\n"
            "- Do not merge facts from different entities, documents, or tables into a single key point.\n"
            "- Keep entity names, departments, companies, products, and numeric values aligned exactly with the source turns.\n"
            "- open_questions should include unresolved tasks or missing data.\n"
            "- long_term_candidates are only for cross-session memory.\n"
            "- long_term_candidates must include only durable user/project context such as goals, preferences, constraints, identity, or long-running project state.\n"
            "- Allowed long_term_categories: project, preference, constraint, goal, identity.\n"
            "- Set importance >= 0.6 only when the memory is truly durable across sessions.\n"
            "- project memories should describe the ongoing system, repository, workflow, or project context, not a one-off business answer topic.\n"
            "- Do not put retrieved business facts, table results, report data, or one-off answers into long_term_candidates.\n"
            "- Do not invent facts."
        )
        user_prompt = (
            f"actor_id={actor_id}\n"
            f"session_id={session_id}\n"
            f"turns={json.dumps(turns, ensure_ascii=False)}\n\n"
            "Return JSON only."
        )

        if provider in {"openai", "openai-compatible", "bailian", "zhipu"}:
            return self._openai_compatible_json(provider, system_prompt, user_prompt)
        if provider == "gemini":
            return self._gemini_json(system_prompt, user_prompt)
        return {}

    @staticmethod
    def _heuristic_summary_payload(turns: list[dict[str, Any]]) -> dict[str, Any]:
        if not turns:
            return {"summary": "", "key_points": [], "open_questions": [], "long_term_candidates": []}
        summary_parts = [f"用户询问：{row.get('user_query', '')}；回答：{row.get('answer', '')}" for row in turns[-3:]]
        key_points = [str(row.get("effective_query", "") or row.get("user_query", "")) for row in turns[-4:]]
        return {
            "summary": " | ".join(summary_parts)[:600],
            "key_points": [item for item in key_points if item][:6],
            "open_questions": [],
            "long_term_candidates": [],
        }

    def _openai_compatible_json(self, provider: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
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
            return {}
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return _extract_json_object(response.choices[0].message.content or "")
        except Exception:
            return {}

    def _openai_compatible_text(self, provider: str, system_prompt: str, user_prompt: str) -> str:
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
            return ""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _gemini_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.settings.gemini_api_key:
            return {}
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(model_name=self.settings.gemini_chat_model)
            response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return _extract_json_object(response.text or "")
        except Exception:
            return {}

    def _gemini_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.settings.gemini_api_key:
            return ""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(model_name=self.settings.gemini_chat_model)
            response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return (response.text or "").strip()
        except Exception:
            return ""


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
