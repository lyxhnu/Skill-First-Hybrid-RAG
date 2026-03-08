from __future__ import annotations

import math
import re
from typing import Iterable

from ..types import QueryConstraintPlan


_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_TOKEN = re.compile(r"[A-Za-z0-9_]{2,}")
_SPLIT_PATTERN = re.compile(r"[，。！？、；：,.!?;:/\\()\[\]{}<>\-\s]+")
_CJK_FILLER_TERMS = (
    "请帮我",
    "帮我",
    "帮忙",
    "请问",
    "分析一下",
    "分析下",
    "看一下",
    "看下",
    "查一下",
    "查下",
    "哪些",
    "哪个",
    "什么",
    "怎么",
    "如何",
    "一下",
    "是否",
    "有无",
)
_STOP_TERMS = {
    "帮我",
    "帮忙",
    "请问",
    "一下",
    "哪些",
    "哪个",
    "什么",
    "怎么",
    "如何",
    "是否",
    "有无",
    "这个",
    "那个",
    "一下子",
    "数据",
    "内容",
    "信息",
    "问题",
}


def normalize_text(value: str) -> str:
    value = value.lower().replace("\u3000", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_keywords(query: str) -> list[str]:
    query = normalize_text(query)
    deduped: list[str] = []
    seen: set[str] = set()

    def add_term(term: str) -> None:
        term = normalize_text(term)
        if len(term) < 2 or term in _STOP_TERMS or term in seen:
            return
        seen.add(term)
        deduped.append(term)

    for token in _LATIN_TOKEN.findall(query):
        add_term(token)

    cjk_text = query
    for filler in _CJK_FILLER_TERMS:
        cjk_text = cjk_text.replace(filler, " ")

    for sequence in _CJK_RUN.findall(cjk_text):
        parts = [part for part in _SPLIT_PATTERN.split(sequence) if part]
        if not parts:
            parts = [sequence]
        for part in parts:
            add_term(part)
            if len(part) <= 2:
                continue
            max_size = min(4, len(part))
            for size in range(max_size, 1, -1):
                for offset in range(0, len(part) - size + 1):
                    add_term(part[offset : offset + size])
    return deduped[:24]


def lexical_score(
    query: str,
    text: str,
    extra_terms: Iterable[str] | None = None,
    query_plan: QueryConstraintPlan | None = None,
) -> float:
    plan = query_plan or QueryConstraintPlan(raw_query=query, soft_terms=extract_keywords(query))
    hard_terms = _dedupe_terms(plan.hard_terms)
    soft_terms = _dedupe_terms(plan.soft_terms or extract_keywords(query))

    if extra_terms:
        for term in extra_terms:
            for candidate in extract_keywords(term) or [term]:
                candidate = normalize_text(candidate)
                if not candidate:
                    continue
                if candidate not in hard_terms and candidate not in soft_terms:
                    soft_terms.append(candidate)

    if not hard_terms and not soft_terms:
        return 0.0

    target = normalize_text(text)
    hard_score = 0.0
    soft_score = 0.0
    matched_hard = 0
    matched_soft = 0

    for term in hard_terms:
        hits = target.count(term)
        if hits > 0:
            matched_hard += 1
            hard_score += _term_score(term, hits, hard=True)
            continue
        partial = _hard_term_partial_match(term, target)
        if partial > 0:
            hard_score += partial

    for term in soft_terms:
        hits = target.count(term)
        if hits > 0:
            matched_soft += 1
            soft_score += _term_score(term, hits, hard=False)
            continue
        partial = _soft_term_partial_match(term, target)
        if partial > 0:
            matched_soft += 1
            soft_score += partial

    if hard_terms:
        coverage = matched_hard / len(hard_terms)
        if matched_hard == 0:
            soft_score *= 0.12
        else:
            soft_score *= 0.4 + (0.6 * coverage)
            hard_score += matched_hard * 2.2
            if soft_terms:
                if matched_soft == 0:
                    hard_score *= 0.28
                else:
                    soft_score *= 1.18
                    soft_score += matched_soft * 0.9
    elif soft_terms and matched_soft > 0:
        soft_score += matched_soft * 0.25

    return hard_score + soft_score


def _term_score(term: str, hits: int, *, hard: bool) -> float:
    base = 2.1 if hard else 1.0
    growth = 0.22 if hard else 0.16
    weight = base + min(len(term), 8) * growth
    return weight * (1.0 + math.log1p(max(0, hits - 1)))


def _hard_term_partial_match(term: str, target: str) -> float:
    fragments = [token for token in extract_keywords(term) if len(token) >= 2]
    if len(fragments) < 2:
        return 0.0
    matched = sum(1 for fragment in fragments if fragment in target)
    if matched == 0:
        return 0.0
    coverage = matched / len(fragments)
    if coverage < 0.6:
        return 0.0
    return 1.1 * coverage


def _soft_term_partial_match(term: str, target: str) -> float:
    fragments = [token for token in extract_keywords(term) if len(token) >= 2]
    if len(fragments) < 2:
        return 0.0
    matched = sum(1 for fragment in fragments if fragment in target)
    if matched == 0:
        return 0.0
    coverage = matched / len(fragments)
    if coverage < 0.45:
        return 0.0
    return (0.8 + min(len(term), 8) * 0.08) * coverage


def _dedupe_terms(terms: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for term in terms:
        normalized = normalize_text(term)
        if not normalized or normalized in deduped:
            continue
        deduped.append(normalized)
    return deduped


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    step = chunk_size - overlap
    chunks: list[tuple[int, int, str]] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == text_len:
            break
        start += step
    return chunks
