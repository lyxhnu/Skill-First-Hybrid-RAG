from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Settings
from ..utils.io import read_json, write_json
from ..utils.text import normalize_text


class CustomerServiceFeedbackManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store_path = self.settings.feedback_dir / "customer_service_gaps.json"
        self.faq_path = self.settings.knowledge_dir / "E-commerce Data" / "faq.json"
        self._lock = threading.RLock()

    def capture_gap(
        self,
        *,
        query: str,
        effective_query: str,
        knowledge_found: bool,
        session_id: str,
        actor_id: str,
        mode: str,
        confidence: float,
        selected_skills: list[str],
        candidate_dirs: list[str],
        candidate_files: list[str],
        evidence_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if knowledge_found:
            return None

        raw_question = str(effective_query or query).strip()
        question_key = self._question_key(raw_question)
        if not question_key:
            return None

        with self._lock:
            payload = self._load_store()
            now = _utcnow()

            resolved = self._find_item(payload["items"], gap_id=question_key, status="resolved")
            if resolved is not None:
                resolved["regression_hits"] = int(resolved.get("regression_hits", 0) or 0) + 1
                resolved["last_regression_at"] = now
                payload["updated_at"] = now
                self._save_store(payload)
                return {
                    "captured": False,
                    "gap_id": resolved["gap_id"],
                    "status": "resolved_regression",
                    "store_path": str(self.store_path),
                }

            existing = self._find_item(payload["items"], gap_id=question_key, status="open")
            if existing is not None:
                existing["hits"] = int(existing.get("hits", 1) or 1) + 1
                existing["last_seen_at"] = now
                existing["last_session_id"] = session_id
                existing["last_actor_id"] = actor_id
                existing["last_mode"] = mode
                existing["last_confidence"] = float(confidence)
                existing["last_candidate_dirs"] = list(candidate_dirs)
                existing["last_candidate_files"] = list(candidate_files)
                existing["last_selected_skills"] = list(selected_skills)
                existing["last_evidence_trace"] = dict(evidence_trace or {})
                payload["updated_at"] = now
                self._save_store(payload)
                return {
                    "captured": True,
                    "gap_id": existing["gap_id"],
                    "status": "open",
                    "hits": existing["hits"],
                    "store_path": str(self.store_path),
                }

            item = {
                "gap_id": question_key,
                "status": "open",
                "created_at": now,
                "updated_at": now,
                "last_seen_at": now,
                "hits": 1,
                "source": "knowledge_unanswered",
                "question": str(query).strip(),
                "effective_query": raw_question,
                "question_key": question_key,
                "session_id": session_id,
                "actor_id": actor_id,
                "last_session_id": session_id,
                "last_actor_id": actor_id,
                "last_mode": mode,
                "last_confidence": float(confidence),
                "last_selected_skills": list(selected_skills),
                "last_candidate_dirs": list(candidate_dirs),
                "last_candidate_files": list(candidate_files),
                "last_evidence_trace": dict(evidence_trace or {}),
            }
            payload["items"].append(item)
            payload["updated_at"] = now
            self._save_store(payload)
            return {
                "captured": True,
                "gap_id": item["gap_id"],
                "status": "open",
                "hits": 1,
                "store_path": str(self.store_path),
            }

    def list_gaps(self, *, status: str = "open", limit: int = 100) -> dict[str, Any]:
        status = (status or "open").strip().lower()
        if status not in {"open", "resolved", "all"}:
            raise ValueError("status must be one of: open, resolved, all")
        limit = max(1, min(int(limit), 500))

        payload = self._load_store()
        items = list(payload["items"])
        if status != "all":
            items = [item for item in items if str(item.get("status", "")).lower() == status]
        items.sort(key=lambda item: str(item.get("last_seen_at") or item.get("updated_at") or ""), reverse=True)
        return {
            "status": status,
            "total": len(items),
            "items": items[:limit],
        }

    def resolve_gap(
        self,
        *,
        gap_id: str,
        answer: str,
        reviewer: str | None = None,
        label: str | None = None,
        question: str | None = None,
        url: str | None = None,
    ) -> dict[str, Any]:
        answer = str(answer or "").strip()
        if not answer:
            raise ValueError("answer must not be empty")

        with self._lock:
            payload = self._load_store()
            item = self._find_item(payload["items"], gap_id=gap_id)
            if item is None:
                raise ValueError(f"gap not found: {gap_id}")

            final_question = str(question or item.get("effective_query") or item.get("question") or "").strip()
            if not final_question:
                raise ValueError("question must not be empty")

            faq_payload, records, container_key = self._load_faq_records()
            faq_entry, created = self._upsert_faq_record(
                records=records,
                question=final_question,
                answer=answer,
                reviewer=reviewer,
                label=label,
                url=url,
            )
            self._save_faq_records(faq_payload, records, container_key)

            now = _utcnow()
            item["status"] = "resolved"
            item["updated_at"] = now
            item["resolved_at"] = now
            item["resolved_by"] = str(reviewer or "").strip()
            item["resolved_answer"] = answer
            item["faq_record_id"] = faq_entry["id"]
            item["faq_path"] = str(self.faq_path)
            payload["updated_at"] = now
            self._save_store(payload)

            return {
                "gap_id": item["gap_id"],
                "status": item["status"],
                "faq_record_id": faq_entry["id"],
                "faq_path": str(self.faq_path),
                "created": created,
                "question": final_question,
                "answer": answer,
            }

    def stats(self) -> dict[str, int]:
        payload = self._load_store()
        open_count = 0
        resolved_count = 0
        for item in payload["items"]:
            if str(item.get("status", "")).lower() == "resolved":
                resolved_count += 1
            else:
                open_count += 1
        return {"open": open_count, "resolved": resolved_count, "total": len(payload["items"])}

    def _load_store(self) -> dict[str, Any]:
        payload = read_json(self.store_path, default={})
        if not isinstance(payload, dict):
            payload = {}
        items = payload.get("items")
        if not isinstance(items, list):
            items = []
        return {
            "version": int(payload.get("version", 1) or 1),
            "updated_at": str(payload.get("updated_at", "")),
            "items": items,
        }

    def _save_store(self, payload: dict[str, Any]) -> None:
        write_json(self.store_path, payload)

    @staticmethod
    def _find_item(items: list[dict[str, Any]], *, gap_id: str, status: str | None = None) -> dict[str, Any] | None:
        target_status = str(status or "").strip().lower()
        for item in items:
            if str(item.get("gap_id", "")) != gap_id:
                continue
            if target_status and str(item.get("status", "")).lower() != target_status:
                continue
            return item
        return None

    def _load_faq_records(self) -> tuple[Any, list[dict[str, Any]], str | None]:
        payload = read_json(self.faq_path, default=[])
        if isinstance(payload, list):
            return payload, payload, None
        if isinstance(payload, dict):
            for key in ("records", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return payload, value, key
            records = []
            payload["records"] = records
            return payload, records, "records"
        records = []
        return records, records, None

    def _save_faq_records(self, payload: Any, records: list[dict[str, Any]], container_key: str | None) -> None:
        if isinstance(payload, dict) and container_key:
            payload[container_key] = records
            write_json(self.faq_path, payload)
            return
        write_json(self.faq_path, records)

    def _upsert_faq_record(
        self,
        *,
        records: list[dict[str, Any]],
        question: str,
        answer: str,
        reviewer: str | None,
        label: str | None,
        url: str | None,
    ) -> tuple[dict[str, Any], bool]:
        question_key = self._question_key(question)
        now = _utcnow()
        for record in records:
            if not isinstance(record, dict):
                continue
            current_question = str(record.get("question", "")).strip()
            if self._question_key(current_question) != question_key:
                continue
            record["answer"] = answer
            record["label"] = str(label or record.get("label") or "ai_customer_service").strip()
            if url is not None:
                record["url"] = str(url).strip()
            record["updated_at"] = now
            record["source"] = "human_feedback"
            if reviewer:
                record["reviewer"] = str(reviewer).strip()
            return record, False

        record_id = hashlib.sha256(question_key.encode("utf-8")).hexdigest()[:16]
        entry = {
            "id": record_id,
            "label": str(label or "ai_customer_service").strip(),
            "question": question,
            "answer": answer,
            "source": "human_feedback",
            "created_at": now,
            "updated_at": now,
        }
        if reviewer:
            entry["reviewer"] = str(reviewer).strip()
        if url:
            entry["url"] = str(url).strip()
        records.append(entry)
        return entry, True

    @staticmethod
    def _question_key(question: str) -> str:
        normalized = normalize_text(question)
        normalized = "".join(char for char in normalized if char.isalnum() or "\u4e00" <= char <= "\u9fff")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()
