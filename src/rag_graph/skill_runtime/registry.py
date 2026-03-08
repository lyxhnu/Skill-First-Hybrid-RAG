from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from ..utils.io import read_text_with_fallback
from ..utils.text import lexical_score, normalize_text


@dataclass
class SkillSpec:
    skill_id: str
    name: str
    description: str
    root_path: Path
    skill_md_path: Path
    references: list[Path]
    scripts: list[Path]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "root_path": str(self.root_path),
            "skill_md_path": str(self.skill_md_path),
            "references": [str(path) for path in self.references],
            "scripts": [str(path) for path in self.scripts],
            "metadata": self.metadata,
        }


class SkillRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.skills_root = self.settings.project_root / ".agent" / "skills"
        self._skills: dict[str, SkillSpec] = {}
        self.reload()

    def reload(self) -> None:
        self._skills = self._discover_skills()

    def list_skills(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for _, spec in sorted(self._skills.items(), key=lambda item: item[0])]

    def get(self, skill_id: str) -> SkillSpec | None:
        if skill_id in self._skills:
            return self._skills[skill_id]
        key = normalize_text(skill_id)
        for spec in self._skills.values():
            if normalize_text(spec.name) == key:
                return spec
        return None

    def select_for_query(self, query: str, top_n: int = 3) -> list[str]:
        scored: list[tuple[float, str]] = []
        for skill_id, spec in self._skills.items():
            payload = f"{spec.skill_id}\n{spec.name}\n{spec.description}"
            score = lexical_score(query, payload) + self._intent_boost(query, skill_id)
            scored.append((score, skill_id))
        if not scored:
            return []
        scored.sort(key=lambda row: row[0], reverse=True)
        selected = [skill_id for score, skill_id in scored if score > 0][:top_n]
        if selected:
            return selected
        if "rag-skill" in self._skills:
            return ["rag-skill"]
        return [scored[0][1]]

    def _discover_skills(self) -> dict[str, SkillSpec]:
        if not self.skills_root.exists():
            return {}
        found: dict[str, SkillSpec] = {}
        for directory in sorted(self.skills_root.iterdir()):
            if not directory.is_dir():
                continue
            skill_md = directory / "SKILL.md"
            if not skill_md.exists():
                continue
            text = read_text_with_fallback(skill_md)
            frontmatter = _parse_frontmatter(text)
            name = frontmatter.get("name", directory.name)
            description = frontmatter.get("description", _first_nonempty_line(text))
            references = sorted(path for path in (directory / "references").glob("*") if path.is_file()) if (directory / "references").exists() else []
            scripts = sorted(path for path in (directory / "scripts").glob("*") if path.is_file()) if (directory / "scripts").exists() else []
            found[directory.name] = SkillSpec(
                skill_id=directory.name,
                name=name,
                description=description,
                root_path=directory.resolve(),
                skill_md_path=skill_md.resolve(),
                references=[path.resolve() for path in references],
                scripts=[path.resolve() for path in scripts],
                metadata={"frontmatter": frontmatter},
            )
        return found

    @staticmethod
    def _intent_boost(query: str, skill_id: str) -> float:
        normalized = normalize_text(query)
        boost = 0.0
        if skill_id == "rag-skill":
            rag_hints = ["知识库", "检索", "问答", "rag", "pdf", "excel", "data_structure"]
            if any(hint in normalized for hint in rag_hints):
                boost += 2.2
        if skill_id == "skill-creator":
            creator_hints = ["创建skill", "新建skill", "skill模板", "package_skill", "quick_validate", "技能开发"]
            if any(hint in normalized for hint in creator_hints):
                boost += 2.4
        return boost


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    payload: dict[str, str] = {}
    for line in lines[1:]:
        line = line.strip()
        if line == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""

