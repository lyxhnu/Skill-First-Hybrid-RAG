from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..service import RAGService


@dataclass
class EvalCase:
    query: str
    expected_keyword: str


def default_eval_set() -> list[EvalCase]:
    return [
        EvalCase(query="XSS 攻击的防护措施有哪些？", expected_keyword="XSS"),
        EvalCase(query="库存数据里哪些商品库存不足？", expected_keyword="库存"),
        EvalCase(query="2026年AI Agent的关键趋势是什么？", expected_keyword="AI Agent"),
    ]


def evaluate_service(service: RAGService, mode: str = "hybrid") -> dict[str, Any]:
    cases = default_eval_set()
    passed = 0
    results: list[dict[str, Any]] = []
    for case in cases:
        response = service.query(query=case.query, mode=mode, top_k=8)
        answer = response.get("answer", "")
        hit = case.expected_keyword.lower() in answer.lower()
        if hit:
            passed += 1
        results.append(
            {
                "query": case.query,
                "expected_keyword": case.expected_keyword,
                "pass": hit,
                "citations": len(response.get("citations", [])),
            }
        )
    return {
        "mode": mode,
        "total": len(cases),
        "passed": passed,
        "pass_rate": (passed / len(cases)) if cases else 0.0,
        "results": results,
    }
