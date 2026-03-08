from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    project_root: Path = Field(default=DEFAULT_PROJECT_ROOT)
    knowledge_dir: Path = Field(default=DEFAULT_PROJECT_ROOT / "knowledge")
    storage_dir: Path = Field(default=DEFAULT_PROJECT_ROOT / "storage")
    memory_dir: Path = Field(default=DEFAULT_PROJECT_ROOT / "storage" / "memory")

    chunk_size: int = 900
    chunk_overlap: int = 120

    max_skill_iterations: int = 5
    skill_min_hits: int = 3
    skill_confidence_threshold: float = 1.2

    default_top_k: int = 8
    max_top_k: int = 25
    vector_weight: float = 0.85
    skill_weight: float = 1.0
    memory_window_turns: int = 4
    memory_summary_trigger_turns: int = 6
    memory_summary_block_turns: int = 6
    memory_summary_top_k: int = 3
    memory_recall_turns: int = 4
    long_term_memory_top_k: int = 5

    chat_provider: str = "zhipu"
    chat_model: str = "glm-5"
    embed_provider: str = "bailian"
    embed_model: str = "text-embedding-v4"
    rerank_provider: str = "bailian"
    rerank_model: str = "qwen3-rerank"
    local_embedding_dim: int = 1536

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    bailian_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DASHSCOPE_API_KEY", "RAG_BAILIAN_API_KEY"),
    )
    bailian_base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    bailian_chat_model: str = "qwen-plus"
    bailian_embedding_model: str = "text-embedding-v4"
    bailian_rerank_model: str = "qwen3-rerank"
    bailian_rerank_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"

    zhipu_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ZHIPU_API_KEY", "RAG_ZHIPU_API_KEY"),
    )
    zhipu_base_url: str | None = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_chat_model: str = "glm-4-flash"
    zhipu_embedding_model: str = "embedding-3"

    anthropic_api_key: str | None = None
    anthropic_chat_model: str = "claude-3-5-sonnet-latest"

    gemini_api_key: str | None = None
    gemini_chat_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "models/text-embedding-004"

    @property
    def parsed_chunks_path(self) -> Path:
        return self.storage_dir / "parsed" / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.storage_dir / "metadata" / "manifest.json"

    @property
    def vector_index_path(self) -> Path:
        return self.storage_dir / "vector" / "embedding_index.pkl"

    def ensure_storage_dirs(self) -> None:
        (self.storage_dir / "parsed").mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "vector").mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "actors").mkdir(parents=True, exist_ok=True)
