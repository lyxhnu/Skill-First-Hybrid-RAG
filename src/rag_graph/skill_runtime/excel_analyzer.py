from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Settings
from ..types import QueryConstraintPlan, query_plan_from_dict
from ..utils.io import read_text_with_fallback
from ..utils.text import lexical_score, normalize_text


DEFAULT_MAX_WORKBOOKS = 3


class ExcelStructuredAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.skill_root = self.settings.project_root / ".agent" / "skills" / "rag-skill"
        self.excel_read_ref = self.skill_root / "references" / "excel_reading.md"
        self.excel_analysis_ref = self.skill_root / "references" / "excel_analysis.md"
        self._guidance_cache: tuple[str, str] | None = None

    def analyze(
        self,
        query: str,
        candidate_files: list[str],
        top_k: int,
        query_plan: QueryConstraintPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        excel_files = [Path(source_path) for source_path in candidate_files if source_path.lower().endswith(".xlsx")]
        if not excel_files:
            return []

        guidance = self._load_guidance()
        plan = _coerce_query_plan(query, query_plan)
        results: list[dict[str, Any]] = []

        for path in excel_files[:DEFAULT_MAX_WORKBOOKS]:
            if not path.exists():
                continue
            workbook = self._load_workbook(path)
            if not workbook:
                continue
            analysis_plan = self._plan_workbook_analysis(
                query=query,
                path=path,
                workbook=workbook,
                query_plan=plan,
                guidance=guidance,
                top_k=top_k,
            )
            if not analysis_plan:
                continue
            results.extend(
                self._execute_analysis_plan(
                    query=query,
                    path=path,
                    workbook=workbook,
                    analysis_plan=analysis_plan,
                    query_plan=plan,
                    top_k=top_k,
                    guidance=guidance,
                )
            )

        results.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return results[: max(top_k * 2, top_k)]

    def _load_guidance(self) -> tuple[str, str]:
        if self._guidance_cache is not None:
            return self._guidance_cache
        if not self.excel_read_ref.exists() or not self.excel_analysis_ref.exists():
            raise RuntimeError("Missing required Excel reference files under .agent/skills/rag-skill/references/")
        self._guidance_cache = (
            read_text_with_fallback(self.excel_read_ref),
            read_text_with_fallback(self.excel_analysis_ref),
        )
        return self._guidance_cache

    @staticmethod
    def _load_workbook(path: Path) -> dict[str, pd.DataFrame]:
        try:
            workbook = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        except Exception:
            return {}
        return {str(sheet_name): frame for sheet_name, frame in workbook.items() if frame is not None and not frame.empty}

    def _plan_workbook_analysis(
        self,
        *,
        query: str,
        path: Path,
        workbook: dict[str, pd.DataFrame],
        query_plan: QueryConstraintPlan | None,
        guidance: tuple[str, str],
        top_k: int,
    ) -> dict[str, Any] | None:
        workbook_preview = self._preview_workbook(workbook)
        payload = self._call_planner_model(
            query=query,
            path=path,
            workbook_preview=workbook_preview,
            query_plan=query_plan,
            guidance=guidance,
            top_k=top_k,
        )
        if not payload:
            return None
        return self._sanitize_analysis_plan(payload, workbook)

    def _call_planner_model(
        self,
        *,
        query: str,
        path: Path,
        workbook_preview: dict[str, Any],
        query_plan: QueryConstraintPlan | None,
        guidance: tuple[str, str],
        top_k: int,
    ) -> dict[str, Any]:
        provider = self.settings.chat_provider.lower().strip()
        system_prompt = self._planner_system_prompt(guidance)
        user_prompt = self._planner_user_prompt(
            query=query,
            path=path,
            workbook_preview=workbook_preview,
            query_plan=query_plan,
            top_k=top_k,
        )

        if provider in {"openai", "openai-compatible", "bailian", "zhipu"}:
            return self._openai_compatible_plan(provider, system_prompt, user_prompt)
        if provider == "gemini":
            return self._gemini_plan(system_prompt, user_prompt)
        return {}

    def _openai_compatible_plan(self, provider: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
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

    def _gemini_plan(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
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

    @staticmethod
    def _preview_workbook(workbook: dict[str, pd.DataFrame]) -> dict[str, Any]:
        sheets: list[dict[str, Any]] = []
        for sheet_name, frame in workbook.items():
            preview_rows: list[dict[str, str]] = []
            preview_frame = frame.fillna("").head(3)
            for _, row in preview_frame.iterrows():
                payload: dict[str, str] = {}
                for column, value in row.to_dict().items():
                    text = ExcelStructuredAnalyzer._preview_value(value)
                    if text:
                        payload[str(column)] = text
                if payload:
                    preview_rows.append(payload)
            sheets.append(
                {
                    "sheet_name": sheet_name,
                    "row_count": int(len(frame)),
                    "columns": [str(column) for column in frame.columns],
                    "sample_rows": preview_rows,
                }
            )
        return {"sheets": sheets}

    def _sanitize_analysis_plan(
        self, payload: dict[str, Any], workbook: dict[str, pd.DataFrame]
    ) -> dict[str, Any] | None:
        if payload.get("relevant") is False:
            return None

        sheet_name = self._resolve_sheet_name(payload.get("sheet_name"), workbook)
        if not sheet_name:
            return None
        frame = workbook[sheet_name]
        columns = [str(column) for column in frame.columns]

        operation = _sanitize_scalar(payload.get("operation"), default="none")
        if operation not in {"filter", "extreme", "aggregate", "group_aggregate"}:
            return None

        plan = {
            "sheet_name": sheet_name,
            "operation": operation,
            "filters": self._sanitize_filters(payload.get("filters", []), columns),
            "metric_column": self._resolve_column_name(payload.get("metric_column"), columns),
            "aggregate": _sanitize_scalar(payload.get("aggregate"), default="none"),
            "group_by": self._sanitize_column_list(payload.get("group_by", []), columns),
            "sort": self._sanitize_sort(payload.get("sort", []), columns),
            "select_columns": self._sanitize_column_list(payload.get("select_columns", []), columns),
            "limit": self._sanitize_limit(payload.get("limit")),
            "reason": str(payload.get("reason", "")).strip(),
        }

        if not plan["select_columns"]:
            plan["select_columns"] = self._default_select_columns(columns, plan)
        if (
            plan["operation"] == "filter"
            and not plan["filters"]
            and not plan["sort"]
            and not plan["metric_column"]
            and not plan["group_by"]
        ):
            return None
        if plan["operation"] == "extreme" and not plan["metric_column"]:
            return None
        if plan["operation"] == "aggregate" and plan["aggregate"] != "count" and not plan["metric_column"]:
            return None
        if plan["operation"] == "group_aggregate" and (not plan["group_by"] or not plan["aggregate"]):
            return None
        return plan

    def _execute_analysis_plan(
        self,
        *,
        query: str,
        path: Path,
        workbook: dict[str, pd.DataFrame],
        analysis_plan: dict[str, Any],
        query_plan: QueryConstraintPlan | None,
        top_k: int,
        guidance: tuple[str, str],
    ) -> list[dict[str, Any]]:
        frame = workbook[analysis_plan["sheet_name"]].copy()
        filtered = self._apply_filters(frame, analysis_plan["filters"])
        if filtered.empty:
            return []

        operation = analysis_plan["operation"]
        if operation == "filter":
            return self._execute_filter_plan(path, filtered, analysis_plan, query, query_plan, top_k, guidance)
        if operation == "extreme":
            return self._execute_extreme_plan(path, filtered, analysis_plan, query, query_plan, top_k, guidance)
        if operation == "aggregate":
            return self._execute_aggregate_plan(path, filtered, analysis_plan, query, query_plan, guidance)
        if operation == "group_aggregate":
            return self._execute_group_aggregate_plan(path, filtered, analysis_plan, query, query_plan, top_k, guidance)
        return []

    def _execute_filter_plan(
        self,
        path: Path,
        frame: pd.DataFrame,
        plan: dict[str, Any],
        query: str,
        query_plan: QueryConstraintPlan | None,
        top_k: int,
        guidance: tuple[str, str],
    ) -> list[dict[str, Any]]:
        ordered = self._apply_sort(frame, plan["sort"])
        limit = max(1, min(int(plan["limit"] or top_k), top_k))
        selected = ordered.head(limit)
        summary_rows = [self._row_summary(row, plan["select_columns"]) for _, row in selected.iterrows()]
        summary = (
            f"Excel 分析结果：在文件 {path.name} 的工作表 {plan['sheet_name']} 中，"
            f"按分析计划筛选到 {len(ordered)} 行匹配数据。"
        )
        if summary_rows:
            summary += f" 前 {len(summary_rows)} 条为：{'；'.join(summary_rows)}。"
        return self._build_tabular_evidence(
            path=path,
            frame=selected,
            plan=plan,
            query=query,
            query_plan=query_plan,
            guidance=guidance,
            summary_content=summary,
            analysis_type="generic_filter",
        )

    def _execute_extreme_plan(
        self,
        path: Path,
        frame: pd.DataFrame,
        plan: dict[str, Any],
        query: str,
        query_plan: QueryConstraintPlan | None,
        top_k: int,
        guidance: tuple[str, str],
    ) -> list[dict[str, Any]]:
        metric_column = plan["metric_column"]
        if not metric_column:
            return []
        numeric = pd.to_numeric(frame[metric_column], errors="coerce")
        working = frame.loc[numeric.notna()].copy()
        if working.empty:
            return []
        working["__metric__"] = numeric.loc[working.index]
        ascending = plan.get("aggregate") == "min"
        working = working.sort_values("__metric__", ascending=ascending)
        limit = max(1, min(int(plan["limit"] or 1), top_k))
        selected = working.head(limit)
        first_row = selected.iloc[0]
        direction = "最低" if ascending else "最高"
        summary = (
            f"Excel 分析结果：在文件 {path.name} 的工作表 {plan['sheet_name']} 中，"
            f"按列 {metric_column} 排序后，{direction}值对应的记录是："
            f"{self._row_summary(first_row, plan['select_columns'])}。"
        )
        return self._build_tabular_evidence(
            path=path,
            frame=selected.drop(columns=["__metric__"], errors="ignore"),
            plan=plan,
            query=query,
            query_plan=query_plan,
            guidance=guidance,
            summary_content=summary,
            analysis_type=f"generic_{plan.get('aggregate', 'extreme')}",
        )

    def _execute_aggregate_plan(
        self,
        path: Path,
        frame: pd.DataFrame,
        plan: dict[str, Any],
        query: str,
        query_plan: QueryConstraintPlan | None,
        guidance: tuple[str, str],
    ) -> list[dict[str, Any]]:
        aggregate = plan["aggregate"]
        metric_column = plan["metric_column"]
        value: Any
        if aggregate == "count":
            value = int(len(frame))
        else:
            series = pd.to_numeric(frame[metric_column], errors="coerce")
            series = series.dropna()
            if series.empty:
                return []
            if aggregate == "sum":
                value = float(series.sum())
            elif aggregate == "avg":
                value = float(series.mean())
            elif aggregate == "min":
                value = float(series.min())
            elif aggregate == "max":
                value = float(series.max())
            else:
                return []

        summary = (
            f"Excel 分析结果：在文件 {path.name} 的工作表 {plan['sheet_name']} 中，"
            f"{aggregate} 结果为 {self._format_number(value)}。"
        )
        return [
            self._build_evidence(
                path=path,
                location={"sheet_name": plan["sheet_name"], "analysis": aggregate},
                content=summary,
                score=8.55 + self._score_bonus(query, path, frame.columns, query_plan),
                metadata=self._analysis_metadata(plan, guidance, aggregate),
            )
        ]

    def _execute_group_aggregate_plan(
        self,
        path: Path,
        frame: pd.DataFrame,
        plan: dict[str, Any],
        query: str,
        query_plan: QueryConstraintPlan | None,
        top_k: int,
        guidance: tuple[str, str],
    ) -> list[dict[str, Any]]:
        group_by = plan["group_by"]
        aggregate = plan["aggregate"]
        metric_column = plan["metric_column"]
        if aggregate == "count":
            grouped = frame.groupby(group_by).size().reset_index(name="value")
        else:
            series = pd.to_numeric(frame[metric_column], errors="coerce")
            working = frame.loc[series.notna()].copy()
            if working.empty:
                return []
            working["__metric__"] = series.loc[working.index]
            grouped_series = working.groupby(group_by)["__metric__"]
            if aggregate == "sum":
                grouped = grouped_series.sum().reset_index(name="value")
            elif aggregate == "avg":
                grouped = grouped_series.mean().reset_index(name="value")
            elif aggregate == "min":
                grouped = grouped_series.min().reset_index(name="value")
            elif aggregate == "max":
                grouped = grouped_series.max().reset_index(name="value")
            else:
                return []

        sort_spec = plan["sort"] or [{"column": "value", "ascending": False}]
        ordered = self._apply_sort(grouped, sort_spec)
        limit = max(1, min(int(plan["limit"] or top_k), top_k))
        selected = ordered.head(limit)
        summary_rows = [self._row_summary(row, [*group_by, "value"]) for _, row in selected.iterrows()]
        summary = (
            f"Excel 分析结果：在文件 {path.name} 的工作表 {plan['sheet_name']} 中，"
            f"按 {', '.join(group_by)} 分组并执行 {aggregate} 聚合。"
        )
        if summary_rows:
            summary += f" 前 {len(summary_rows)} 条为：{'；'.join(summary_rows)}。"
        plan_for_rows = dict(plan)
        plan_for_rows["select_columns"] = [*group_by, "value"]
        return self._build_tabular_evidence(
            path=path,
            frame=selected,
            plan=plan_for_rows,
            query=query,
            query_plan=query_plan,
            guidance=guidance,
            summary_content=summary,
            analysis_type=f"group_{aggregate}",
        )

    def _build_tabular_evidence(
        self,
        *,
        path: Path,
        frame: pd.DataFrame,
        plan: dict[str, Any],
        query: str,
        query_plan: QueryConstraintPlan | None,
        guidance: tuple[str, str],
        summary_content: str,
        analysis_type: str,
    ) -> list[dict[str, Any]]:
        score_bonus = self._score_bonus(query, path, frame.columns, query_plan)
        evidence = [
            self._build_evidence(
                path=path,
                location={"sheet_name": plan["sheet_name"], "analysis": analysis_type},
                content=summary_content,
                score=8.4 + score_bonus,
                metadata=self._analysis_metadata(plan, guidance, analysis_type, result_count=len(frame)),
            )
        ]

        for rank, (row_index, row) in enumerate(frame.head(max(1, min(6, len(frame)))).iterrows(), start=1):
            content = self._row_summary(row, plan["select_columns"])
            evidence.append(
                self._build_evidence(
                    path=path,
                    location={"sheet_name": plan["sheet_name"], "row_index": int(row_index) + 2},
                    content=content,
                    score=7.55 - rank * 0.08 + score_bonus,
                    metadata=self._analysis_metadata(plan, guidance, f"{analysis_type}_row"),
                )
            )
        return evidence

    @staticmethod
    def _apply_sort(frame: pd.DataFrame, sort_spec: list[dict[str, Any]]) -> pd.DataFrame:
        if frame.empty or not sort_spec:
            return frame
        by: list[str] = []
        ascending: list[bool] = []
        for item in sort_spec:
            column = str(item.get("column", "")).strip()
            if not column or column not in frame.columns:
                continue
            by.append(column)
            ascending.append(bool(item.get("ascending", True)))
        if not by:
            return frame
        return frame.sort_values(by=by, ascending=ascending)

    def _apply_filters(self, frame: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
        working = frame.copy()
        for filter_item in filters:
            working = self._apply_single_filter(working, filter_item)
            if working.empty:
                break
        return working

    def _apply_single_filter(self, frame: pd.DataFrame, filter_item: dict[str, Any]) -> pd.DataFrame:
        column = str(filter_item.get("column", "")).strip()
        op = str(filter_item.get("op", "")).strip()
        value = filter_item.get("value")
        value_from_column = str(filter_item.get("value_from_column", "")).strip()
        if not column or column not in frame.columns or op not in {"==", "!=", ">", ">=", "<", "<=", "contains", "in"}:
            return frame

        series = frame[column]
        if value_from_column:
            if value_from_column not in frame.columns:
                return frame
            left = pd.to_numeric(series, errors="coerce")
            right = pd.to_numeric(frame[value_from_column], errors="coerce")
            mask = self._compare_series(left, right, op)
            return frame.loc[mask.fillna(False)]

        if op == "contains":
            text_series = series.astype(str).str.lower()
            expected = str(value or "").strip().lower()
            if not expected:
                return frame
            mask = text_series.str.contains(expected, na=False) | text_series.map(lambda item: expected in item or item in expected)
            return frame.loc[mask]

        if op == "in":
            if not isinstance(value, list):
                return frame
            candidates = {str(item).strip() for item in value if str(item).strip()}
            if not candidates:
                return frame
            return frame.loc[series.astype(str).isin(candidates)]

        scalar = self._coerce_filter_value(series, value)
        if op in {"==", "!="} and scalar is not None and pd.api.types.is_numeric_dtype(series):
            left = pd.to_numeric(series, errors="coerce")
            if op == "==":
                mask = left == scalar
            else:
                mask = left != scalar
            return frame.loc[mask.fillna(False)]

        if op in {">", ">=", "<", "<="}:
            left = pd.to_numeric(series, errors="coerce")
            right = self._coerce_numeric(value)
            if right is None:
                return frame
            mask = self._compare_series(left, right, op)
            return frame.loc[mask.fillna(False)]

        text_series = series.astype(str).str.strip()
        expected = str(value or "").strip()
        if not expected:
            return frame
        if op == "==":
            mask = text_series.eq(expected) | text_series.str.contains(expected, regex=False, na=False) | text_series.map(
                lambda item: expected in item or item in expected
            )
        else:
            mask = ~(text_series.eq(expected) | text_series.str.contains(expected, regex=False, na=False))
        return frame.loc[mask.fillna(False)]

    @staticmethod
    def _compare_series(left: pd.Series, right: Any, op: str) -> pd.Series:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        return pd.Series([False] * len(left), index=left.index)

    @staticmethod
    def _coerce_numeric(value: Any) -> float | None:
        try:
            if value is None or str(value).strip() == "":
                return None
            return float(value)
        except Exception:
            return None

    def _coerce_filter_value(self, series: pd.Series, value: Any) -> Any:
        if pd.api.types.is_numeric_dtype(series):
            return self._coerce_numeric(value)
        if pd.api.types.is_datetime64_any_dtype(series):
            try:
                return pd.to_datetime(value)
            except Exception:
                return None
        return str(value or "").strip()

    def _analysis_metadata(
        self,
        plan: dict[str, Any],
        guidance: tuple[str, str],
        analysis_type: str,
        result_count: int | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "skill_id": "rag-skill",
            "analysis_type": analysis_type,
            "sheet_name": plan["sheet_name"],
            "plan": {
                "operation": plan["operation"],
                "filters": plan["filters"],
                "metric_column": plan["metric_column"],
                "aggregate": plan["aggregate"],
                "group_by": plan["group_by"],
                "sort": plan["sort"],
                "select_columns": plan["select_columns"],
                "limit": plan["limit"],
            },
            "references_loaded": [
                str(self.excel_read_ref),
                str(self.excel_analysis_ref),
            ],
            "reference_hashes": {
                "excel_reading_md": hashlib.sha256(guidance[0].encode("utf-8")).hexdigest(),
                "excel_analysis_md": hashlib.sha256(guidance[1].encode("utf-8")).hexdigest(),
            },
        }
        if result_count is not None:
            metadata["result_count"] = int(result_count)
        return metadata

    def _score_bonus(
        self,
        query: str,
        path: Path,
        columns: Any,
        query_plan: QueryConstraintPlan | None,
    ) -> float:
        column_text = " ".join(str(column) for column in columns)
        return min(2.0, lexical_score(query, f"{path.name} {column_text}", query_plan=query_plan) * 0.12)

    @staticmethod
    def _preview_value(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip()
        if len(text) > 60:
            return text[:60] + "..."
        return text

    @staticmethod
    def _resolve_sheet_name(value: Any, workbook: dict[str, pd.DataFrame]) -> str:
        if not workbook:
            return ""
        raw = str(value or "").strip()
        if raw in workbook:
            return raw
        normalized = normalize_text(raw)
        for sheet_name in workbook:
            if normalize_text(sheet_name) == normalized:
                return sheet_name
        if len(workbook) == 1:
            return next(iter(workbook))
        return ""

    @staticmethod
    def _resolve_column_name(value: Any, columns: list[str]) -> str:
        raw = str(value or "").strip()
        if raw in columns:
            return raw
        normalized = normalize_text(raw)
        for column in columns:
            if normalize_text(column) == normalized:
                return column
        return ""

    def _sanitize_column_list(self, values: Any, columns: list[str]) -> list[str]:
        if not isinstance(values, list):
            return []
        result: list[str] = []
        for value in values:
            column = self._resolve_column_name(value, columns)
            if column and column not in result:
                result.append(column)
        return result

    def _sanitize_filters(self, values: Any, columns: list[str]) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []
        result: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            column = self._resolve_column_name(item.get("column"), columns)
            op = str(item.get("op", "")).strip()
            value_from_column = self._resolve_column_name(item.get("value_from_column"), columns)
            if not column or op not in {"==", "!=", ">", ">=", "<", "<=", "contains", "in"}:
                continue
            payload = {"column": column, "op": op}
            literal_value = item.get("value")
            inferred_column_value = self._resolve_column_name(literal_value, columns) if not value_from_column else ""
            if value_from_column:
                payload["value_from_column"] = value_from_column
            elif inferred_column_value and op in {">", ">=", "<", "<=", "==", "!="}:
                payload["value_from_column"] = inferred_column_value
            elif "value" in item:
                payload["value"] = literal_value
            else:
                continue
            result.append(payload)
        return result

    def _sanitize_sort(self, values: Any, columns: list[str]) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []
        result: list[dict[str, Any]] = []
        augmented_columns = [*columns, "value"]
        for item in values:
            if not isinstance(item, dict):
                continue
            column = self._resolve_column_name(item.get("column"), augmented_columns)
            if not column:
                continue
            result.append({"column": column, "ascending": bool(item.get("ascending", True))})
        return result

    @staticmethod
    def _sanitize_limit(value: Any) -> int:
        try:
            limit = int(value)
        except Exception:
            return 5
        return max(1, min(limit, 20))

    @staticmethod
    def _default_select_columns(columns: list[str], plan: dict[str, Any]) -> list[str]:
        ordered: list[str] = []
        for key in [plan.get("metric_column"), *plan.get("group_by", [])]:
            if key and key in columns and key not in ordered:
                ordered.append(key)
        for candidate in ("name", "product", "department", "title", "sku", "warehouse", "city"):
            if candidate in columns and candidate not in ordered:
                ordered.insert(0, candidate)
        return ordered[:6] if ordered else columns[: min(6, len(columns))]

    @staticmethod
    def _row_summary(row: pd.Series, columns: list[str]) -> str:
        parts: list[str] = []
        for column in columns:
            if column not in row.index:
                continue
            value = row[column]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if not text:
                continue
            parts.append(f"{column}: {text}")
        return " | ".join(parts) if parts else "无可展示字段"

    @staticmethod
    def _format_number(value: Any) -> str:
        try:
            number = float(value)
        except Exception:
            return str(value)
        if abs(number - round(number)) <= 1e-9:
            return str(int(round(number)))
        return f"{number:.2f}"

    @staticmethod
    def _planner_system_prompt(guidance: tuple[str, str]) -> str:
        excel_reading, excel_analysis = guidance
        return (
            "You are the Excel analysis planner for the rag-skill.\n"
            "You MUST read and follow the two skill references below before planning.\n"
            "Return JSON only.\n\n"
            "[excel_reading.md]\n"
            f"{excel_reading}\n\n"
            "[excel_analysis.md]\n"
            f"{excel_analysis}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "relevant": true,\n'
            '  "sheet_name": "exact sheet name from preview",\n'
            '  "operation": "filter|extreme|aggregate|group_aggregate|none",\n'
            '  "filters": [{"column":"exact column","op":"==|!=|>|>=|<|<=|contains|in","value":"literal","value_from_column":"exact other column when comparing two columns"}],\n'
            '  "metric_column": "exact numeric column name or empty",\n'
            '  "aggregate": "none|max|min|sum|avg|count",\n'
            '  "group_by": ["exact column names"],\n'
            '  "sort": [{"column":"exact column name","ascending":true}],\n'
            '  "select_columns": ["exact column names"],\n'
            '  "limit": 1,\n'
            '  "reason": "short explanation"\n'
            "}\n\n"
            "Rules:\n"
            "- Use exact sheet names and exact column names from the workbook preview.\n"
            "- For highest/lowest questions, use operation=extreme and aggregate=max/min.\n"
            "- If the question combines an entity filter with a highest/lowest/ranking condition, filter the entity first and then use operation=extreme on the ranking metric.\n"
            "- For row filtering questions, use operation=filter.\n"
            "- For totals or averages, use operation=aggregate.\n"
            "- For per-group summaries, use operation=group_aggregate.\n"
            "- Use value_from_column when the condition compares two columns, such as stock_on_hand < reorder_level.\n"
            "- Example: if the workbook has columns name, department, base_salary and the question is '工资最高的员工罗凯在哪个部门工作', choose filters on name=罗凯 and operation=extreme with metric_column=base_salary, aggregate=max, limit=1.\n"
            "- If the workbook is not relevant, return {\"relevant\": false}.\n"
            "- Do not invent columns, values, or sheet names."
        )

    @staticmethod
    def _planner_user_prompt(
        *,
        query: str,
        path: Path,
        workbook_preview: dict[str, Any],
        query_plan: QueryConstraintPlan | None,
        top_k: int,
    ) -> str:
        return (
            f"Question:\n{query}\n\n"
            f"Query plan:\n{_query_plan_to_prompt(query_plan)}\n\n"
            f"Workbook file:\n{path.name}\n\n"
            f"Workbook preview:\n{json.dumps(workbook_preview, ensure_ascii=False)}\n\n"
            f"Top k:\n{top_k}\n\n"
            "Return JSON only."
        )

    @staticmethod
    def _build_evidence(
        *,
        path: Path,
        location: dict[str, Any],
        content: str,
        score: float,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        digest = hashlib.sha256()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(location).encode("utf-8"))
        digest.update(content.encode("utf-8"))
        return {
            "evidence_id": digest.hexdigest(),
            "source_path": str(path.resolve()),
            "file_type": "xlsx",
            "location": location,
            "content": content,
            "retrieval_source": "skill:rag-skill:excel-analysis",
            "score": float(score),
            "domain": "structured-analysis",
            "metadata": metadata,
        }


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


def _sanitize_scalar(value: Any, default: str) -> str:
    normalized = normalize_text(str(value or ""))
    return normalized or default


def _query_plan_to_prompt(query_plan: QueryConstraintPlan | None) -> str:
    if query_plan is None:
        return "hard_terms=[]; soft_terms=[]; intent=lookup; answer_shape=unknown"
    return (
        f"hard_terms={query_plan.hard_terms}; "
        f"soft_terms={query_plan.soft_terms}; "
        f"intent={query_plan.intent}; "
        f"answer_shape={query_plan.answer_shape}"
    )


def _coerce_query_plan(
    query: str,
    query_plan: QueryConstraintPlan | dict[str, Any] | None,
) -> QueryConstraintPlan | None:
    if query_plan is None:
        return None
    if isinstance(query_plan, QueryConstraintPlan):
        return query_plan
    return query_plan_from_dict({"raw_query": query, **query_plan})
