from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

from ..config import Settings
from ..utils.io import iter_jsonl, read_json, read_text_with_fallback, write_json, write_jsonl
from ..utils.text import chunk_text

SUPPORTED_FILE_TYPES = {".md", ".txt", ".pdf", ".xlsx", ".json"}


class IngestEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_storage_dirs()
        self.skill_root = self.settings.project_root / ".agent" / "skills" / "rag-skill"
        self.pdf_ref = self.skill_root / "references" / "pdf_reading.md"
        self.excel_read_ref = self.skill_root / "references" / "excel_reading.md"
        self.excel_analysis_ref = self.skill_root / "references" / "excel_analysis.md"
        self._reference_cache: dict[str, dict[str, Any]] = {}

    def ingest(self, force: bool = False) -> dict[str, Any]:
        scanned_files = self._scan_knowledge_files()
        old_manifest = read_json(self.settings.manifest_path, default={"files": {}})
        old_files: dict[str, str] = old_manifest.get("files", {})
        new_files = {path: self._sha256_file(Path(path)) for path in scanned_files}

        unchanged_files = {
            path
            for path, file_hash in new_files.items()
            if not force and old_files.get(path) == file_hash
        }
        old_chunks = list(iter_jsonl(self.settings.parsed_chunks_path))
        kept_chunks = [chunk for chunk in old_chunks if chunk["source_path"] in unchanged_files]

        parsed_chunks: list[dict[str, Any]] = []
        changed_files = [path for path in scanned_files if path not in unchanged_files]
        for source_path in changed_files:
            parsed_chunks.extend(self._parse_file(Path(source_path)))

        all_chunks = kept_chunks + parsed_chunks
        write_jsonl(self.settings.parsed_chunks_path, all_chunks)

        manifest = {
            "updated_at": datetime.now(UTC).isoformat(),
            "files": new_files,
            "chunk_count": len(all_chunks),
            "file_count": len(scanned_files),
            "changed_files": len(changed_files),
        }
        write_json(self.settings.manifest_path, manifest)
        return manifest

    def load_chunks(self) -> list[dict[str, Any]]:
        return list(iter_jsonl(self.settings.parsed_chunks_path))

    def _scan_knowledge_files(self) -> list[str]:
        root = self.settings.knowledge_dir
        if not root.exists():
            return []
        files: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_FILE_TYPES:
                continue
            files.append(str(path.resolve()))
        files.sort()
        return files

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(8192), b""):
                digest.update(block)
        return digest.hexdigest()

    def _parse_file(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            return self._parse_text_file(path, file_type=suffix[1:])
        if suffix == ".pdf":
            return self._parse_pdf_file(path)
        if suffix == ".xlsx":
            return self._parse_excel_file(path)
        if suffix == ".json":
            return self._parse_json_file(path)
        return []

    def _parse_text_file(self, path: Path, file_type: str) -> list[dict[str, Any]]:
        text = read_text_with_fallback(path)
        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        results: list[dict[str, Any]] = []
        for start, end, payload in chunks:
            location = self._char_offset_to_line_range(text, start, end)
            results.append(self._build_chunk(path, file_type=file_type, content=payload, location=location))
        return results

    def _parse_pdf_file(self, path: Path) -> list[dict[str, Any]]:
        reference_metadata = self._load_pdf_reference_metadata()
        sidecar_txt = path.with_suffix(".txt")
        if sidecar_txt.exists():
            text = read_text_with_fallback(sidecar_txt)
            chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
            results: list[dict[str, Any]] = []
            for start, end, payload in chunks:
                location = self._char_offset_to_line_range(text, start, end)
                results.append(
                    self._build_chunk(
                        path,
                        file_type="pdf",
                        content=payload,
                        location=location,
                        metadata=dict(reference_metadata),
                    )
                )
            return results

        results: list[dict[str, Any]] = []
        try:
            reader = PdfReader(str(path))
            for page_index, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                for _, _, payload in chunk_text(
                    page_text, self.settings.chunk_size, self.settings.chunk_overlap
                ):
                    location = {"page": page_index}
                    results.append(
                        self._build_chunk(
                            path,
                            file_type="pdf",
                            content=payload,
                            location=location,
                            metadata=dict(reference_metadata),
                        )
                    )
        except Exception:
            return []
        return results

    def _parse_excel_file(self, path: Path) -> list[dict[str, Any]]:
        reference_metadata = self._load_excel_reference_metadata()
        results: list[dict[str, Any]] = []
        workbook = pd.read_excel(path, sheet_name=None, dtype=str, engine="openpyxl")
        for sheet_name, dataframe in workbook.items():
            frame = dataframe.fillna("")
            for row_index, row in frame.iterrows():
                row_pairs = [
                    f"{column}: {str(value).strip()}"
                    for column, value in row.to_dict().items()
                    if str(value).strip()
                ]
                if not row_pairs:
                    continue
                content = " | ".join(row_pairs)
                location = {"sheet_name": str(sheet_name), "row_index": int(row_index) + 2}
                results.append(
                    self._build_chunk(
                        path,
                        file_type="xlsx",
                        content=content,
                        location=location,
                        metadata=dict(reference_metadata),
                    )
                )
        return results

    def _parse_json_file(self, path: Path) -> list[dict[str, Any]]:
        text = read_text_with_fallback(path)
        try:
            payload = json.loads(text)
        except Exception:
            # Fallback to text ingestion for non-standard or partially broken JSON.
            return self._parse_text_file(path, file_type="json")

        records = self._coerce_json_records(payload)
        results: list[dict[str, Any]] = []
        for record_index, record in enumerate(records, start=1):
            content, metadata = self._build_json_record_payload(record, record_index)
            if not content.strip():
                continue
            location = {"record_index": record_index}
            question = str(metadata.get("question", "")).strip()
            label = str(metadata.get("label", "")).strip()
            if question:
                location["question"] = question[:120]
            elif label:
                location["label"] = label[:80]
            results.append(
                self._build_chunk(
                    path,
                    file_type="json",
                    content=content,
                    location=location,
                    metadata=metadata,
                )
            )
        return results

    def _build_chunk(
        self,
        path: Path,
        file_type: str,
        content: str,
        location: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_path = str(path.resolve())
        domain = self._infer_domain(path)
        location_text = json.dumps(location, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(
            f"{source_path}|{location_text}|{hashlib.sha256(content.encode('utf-8')).hexdigest()}".encode(
                "utf-8"
            )
        ).hexdigest()
        return {
            "evidence_id": digest,
            "source_path": source_path,
            "file_type": file_type,
            "location": location,
            "content": content,
            "retrieval_source": "ingest",
            "score": 0.0,
            "domain": domain,
            "metadata": metadata or {},
        }

    @staticmethod
    def _coerce_json_records(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("records", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        return [payload]

    def _build_json_record_payload(self, record: Any, record_index: int) -> tuple[str, dict[str, Any]]:
        if isinstance(record, dict):
            question = str(record.get("question", "")).strip()
            answer = str(record.get("answer", "")).strip()
            label = str(record.get("label", "")).strip()
            url = str(record.get("url", "")).strip()
            record_id = str(record.get("id", "")).strip()

            metadata = {
                "record_index": record_index,
                "record_schema": "qa_record",
            }
            if record_id:
                metadata["record_id"] = record_id
            if label:
                metadata["label"] = label
            if question:
                metadata["question"] = question
            if answer:
                metadata["answer"] = answer
            if url:
                metadata["url"] = url

            parts: list[str] = []
            if label:
                parts.append(f"label: {label}")
            if question:
                parts.append(f"question: {question}")
            if answer:
                parts.append(f"answer: {answer}")
            if url:
                parts.append(f"url: {url}")

            if parts:
                return "\n".join(parts), metadata

            flattened = json.dumps(record, ensure_ascii=False, sort_keys=True)
            metadata["record_schema"] = "json_record"
            metadata["raw_record"] = flattened
            return flattened, metadata

        if isinstance(record, (str, int, float, bool)) or record is None:
            value = str(record)
            return value, {"record_index": record_index, "record_schema": "scalar_record"}

        flattened = json.dumps(record, ensure_ascii=False, sort_keys=True)
        return flattened, {"record_index": record_index, "record_schema": "json_record", "raw_record": flattened}

    def _load_pdf_reference_metadata(self) -> dict[str, Any]:
        return self._load_reference_metadata(required=[self.pdf_ref], cache_key="pdf")

    def _load_excel_reference_metadata(self) -> dict[str, Any]:
        return self._load_reference_metadata(
            required=[self.excel_read_ref, self.excel_analysis_ref],
            cache_key="excel",
        )

    def _load_reference_metadata(self, required: list[Path], cache_key: str) -> dict[str, Any]:
        if cache_key in self._reference_cache:
            return dict(self._reference_cache[cache_key])

        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Missing required rag-skill references: {', '.join(missing)}")

        reference_hashes: dict[str, str] = {}
        for path in required:
            content = read_text_with_fallback(path)
            reference_hashes[path.name] = hashlib.sha256(content.encode("utf-8")).hexdigest()

        payload = {
            "references_loaded": [str(path.resolve()) for path in required],
            "reference_hashes": reference_hashes,
        }
        self._reference_cache[cache_key] = payload
        return dict(payload)

    def _infer_domain(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.settings.knowledge_dir.resolve())
            return relative.parts[0] if relative.parts else "knowledge"
        except Exception:
            return "knowledge"

    @staticmethod
    def _char_offset_to_line_range(text: str, start: int, end: int) -> dict[str, int]:
        line_start = text[:start].count("\n") + 1
        line_end = text[:end].count("\n") + 1
        return {"line_start": line_start, "line_end": max(line_end, line_start)}
