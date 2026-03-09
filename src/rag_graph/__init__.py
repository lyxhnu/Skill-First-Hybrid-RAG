"""Skill-first hybrid RAG package."""

__all__ = ["RAGService"]


def __getattr__(name: str):
    if name == "RAGService":
        from .service import RAGService

        return RAGService
    raise AttributeError(f"module 'rag_graph' has no attribute {name!r}")
