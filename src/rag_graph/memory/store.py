from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Settings
from ..utils.io import append_jsonl, iter_jsonl, read_json, write_json


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class MemoryStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = self.settings.memory_dir
        self.sessions_root = self.root / "sessions"
        self.actors_root = self.root / "actors"
        self.settings.ensure_storage_dirs()

    def ensure_session(self, session_id: str, actor_id: str) -> dict[str, Any]:
        meta = self.load_session_meta(session_id)
        if meta:
            if meta.get("actor_id") != actor_id:
                meta["actor_id"] = actor_id
                meta["updated_at"] = utc_now()
                self.save_session_meta(session_id, meta)
            return meta

        created = {
            "session_id": session_id,
            "actor_id": actor_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "next_turn_id": 1,
            "next_summary_id": 1,
            "last_summarized_turn_id": 0,
            "turn_count": 0,
            "summary_count": 0,
        }
        self.save_session_meta(session_id, created)
        return created

    def load_session_meta(self, session_id: str) -> dict[str, Any]:
        return read_json(self._session_meta_path(session_id), default={})

    def save_session_meta(self, session_id: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["updated_at"] = utc_now()
        write_json(self._session_meta_path(session_id), payload)

    def load_turns(self, session_id: str) -> list[dict[str, Any]]:
        return list(iter_jsonl(self._turns_path(session_id)))

    def append_turn(
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
        meta = self.ensure_session(session_id, actor_id)
        turn_id = int(meta.get("next_turn_id", 1))
        turn = {
            "turn_id": turn_id,
            "session_id": session_id,
            "actor_id": actor_id,
            "user_query": user_query,
            "effective_query": effective_query,
            "answer": answer,
            "citations": citations,
            "query_constraints": query_constraints,
            "created_at": utc_now(),
            "path": self.turn_path(session_id, turn_id),
        }
        append_jsonl(self._turns_path(session_id), turn)
        meta["next_turn_id"] = turn_id + 1
        meta["turn_count"] = int(meta.get("turn_count", 0)) + 1
        self.save_session_meta(session_id, meta)
        return turn

    def load_summaries(self, session_id: str) -> list[dict[str, Any]]:
        return list(iter_jsonl(self._summaries_path(session_id)))

    def append_summary(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        meta = self.load_session_meta(session_id)
        summary_id = int(meta.get("next_summary_id", 1))
        summary = dict(payload)
        summary["summary_id"] = f"sb-{summary_id:06d}"
        summary["created_at"] = utc_now()
        append_jsonl(self._summaries_path(session_id), summary)
        meta["next_summary_id"] = summary_id + 1
        meta["summary_count"] = int(meta.get("summary_count", 0)) + 1
        turn_ids = summary.get("turn_ids", [])
        if turn_ids:
            meta["last_summarized_turn_id"] = max(int(value) for value in turn_ids)
        self.save_session_meta(session_id, meta)
        return summary

    def load_long_term(self, actor_id: str) -> list[dict[str, Any]]:
        return list(iter_jsonl(self._long_term_path(actor_id)))

    def append_long_term(self, actor_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        memories = self.load_long_term(actor_id)
        memory_id = f"lt-{len(memories) + 1:06d}"
        row = dict(payload)
        row["memory_id"] = memory_id
        row["created_at"] = utc_now()
        append_jsonl(self._long_term_path(actor_id), row)
        return row

    def resolve_turn_paths(self, paths: list[str]) -> list[dict[str, Any]]:
        grouped: dict[str, set[int]] = {}
        for path in paths:
            session_id, turn_id = self._parse_turn_path(path)
            if not session_id or turn_id is None:
                continue
            grouped.setdefault(session_id, set()).add(turn_id)

        resolved: list[dict[str, Any]] = []
        for session_id, turn_ids in grouped.items():
            for turn in self.load_turns(session_id):
                if int(turn.get("turn_id", -1)) in turn_ids:
                    resolved.append(turn)
        resolved.sort(key=lambda row: (str(row.get("session_id", "")), int(row.get("turn_id", 0))))
        return resolved

    @staticmethod
    def turn_path(session_id: str, turn_id: int) -> str:
        return f"memory://sessions/{session_id}/turns/{turn_id}"

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_root / session_id

    def _session_meta_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "meta.json"

    def _turns_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "turns.jsonl"

    def _summaries_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "summaries.jsonl"

    def _actor_dir(self, actor_id: str) -> Path:
        return self.actors_root / actor_id

    def _long_term_path(self, actor_id: str) -> Path:
        return self._actor_dir(actor_id) / "long_term.jsonl"

    @staticmethod
    def _parse_turn_path(path: str) -> tuple[str, int | None]:
        prefix = "memory://sessions/"
        if not path.startswith(prefix):
            return "", None
        try:
            remainder = path[len(prefix) :]
            session_id, suffix = remainder.split("/turns/", 1)
            return session_id, int(suffix)
        except Exception:
            return "", None
