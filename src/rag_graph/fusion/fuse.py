from __future__ import annotations

from typing import Any

from ..config import Settings


class EvidenceFusion:
    def __init__(self, settings: Settings):
        self.settings = settings

    def fuse(
        self,
        skill_evidence: list[dict[str, Any]],
        vector_evidence: list[dict[str, Any]],
        mode: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if mode == "skill":
            return sorted(skill_evidence, key=lambda item: item["score"], reverse=True)[:top_k]
        if mode == "vector":
            return sorted(vector_evidence, key=lambda item: item["score"], reverse=True)[:top_k]

        merged: dict[str, dict[str, Any]] = {}
        for row in skill_evidence:
            payload = dict(row)
            payload["score"] = float(payload["score"]) * self.settings.skill_weight
            merged[payload["evidence_id"]] = payload

        for row in vector_evidence:
            payload = dict(row)
            payload["score"] = float(payload["score"]) * self.settings.vector_weight
            existing = merged.get(payload["evidence_id"])
            if existing is None or payload["score"] > existing["score"]:
                merged[payload["evidence_id"]] = payload

        combined = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
        return combined[:top_k]

