"""Recursive character text splitter using LangChain.

Provides the default splitter implementation that recursively splits text
using a configurable hierarchy of separators, respecting chunk_size and
chunk_overlap boundaries.
"""

from __future__ import annotations

from typing import Any, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.splitter.base_splitter import BaseSplitter, SplitChunk
from rag_mcp_server.src.libs.splitter.splitter_factory import SplitterFactory


class RecursiveSplitter(BaseSplitter):
    """Recursive character text splitter backed by LangChain.

    Splits text by progressively trying finer-grained separators:
    ``["\\n\\n", "\\n", ". ", "。", " ", ""]``.

    Configuration is read from ``settings.splitter`` (preferred) with
    fallback to ``settings.ingestion``:

    .. code-block:: yaml

        splitter:
          provider: "recursive"
          chunk_size: 1000
          chunk_overlap: 200
          separators: ["\\n\\n", "\\n", ". ", " ", ""]   # optional

    Extra keyword arguments override config values.
    """

    # Default separators ordered from coarsest to finest
    DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", "。", " ", ""]

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the recursive splitter.

        Args:
            settings: Application settings containing splitter and ingestion config.
            **kwargs: Overrides — recognised keys: ``chunk_size``,
                ``chunk_overlap``, ``separators``, ``length_function``.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.settings = settings

        # ── resolve chunk_size ──────────────────────────────────────────
        self.chunk_size: int = kwargs.pop("chunk_size", None)
        if self.chunk_size is None:
            self.chunk_size = self._read_config(
                "chunk_size", default=1000
            )
        self.chunk_size = int(self.chunk_size)

        # ── resolve chunk_overlap ───────────────────────────────────────
        self.chunk_overlap: int = kwargs.pop("chunk_overlap", None)
        if self.chunk_overlap is None:
            self.chunk_overlap = self._read_config(
                "chunk_overlap", default=200
            )
        self.chunk_overlap = int(self.chunk_overlap)

        # ── resolve separators ──────────────────────────────────────────
        separators: Optional[List[str]] = kwargs.pop("separators", None)
        if separators is None:
            raw_seps = self._read_config("separators", default=None)
            if raw_seps and isinstance(raw_seps, list):
                separators = [str(s) for s in raw_seps]
            else:
                separators = list(self.DEFAULT_SEPARATORS)
        self.separators: List[str] = separators

        # ── remaining kwargs go to LangChain ────────────────────────────
        extra: dict[str, Any] = kwargs

        self._lc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=extra.pop("length_function", len),
            **extra,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _read_config(self, key: str, default: Any = None) -> Any:
        """Read a config value from settings.splitter, falling back to
        settings.ingestion, then *default*."""
        splitter_cfg = getattr(self.settings, "splitter", None) or {}
        ingestion_cfg = getattr(self.settings, "ingestion", None) or {}
        # Use explicit None-check to handle falsy-but-valid values like 0
        val = splitter_cfg.get(key)
        if val is not None:
            return val
        val = ingestion_cfg.get(key)
        if val is not None:
            return val
        return default

    # ── public interface ───────────────────────────────────────────────────

    def split(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[SplitChunk]:
        """Split a document text into chunks using recursive character splitting.

        Args:
            text: The full text content of the document to split.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Additional LangChain splitter parameters for this call.

        Returns:
            A list of SplitChunk objects with content, index, and metadata.

        Raises:
            ValueError: If text is empty or not a string.
        """
        self.validate_text(text)

        if not text.strip():
            return []

        try:
            lc_docs = self._lc_splitter.create_documents(
                [text], metadatas=[{}]
            )
        except Exception as exc:
            raise RuntimeError(
                f"RecursiveSplitter failed to split text: {exc}"
            ) from exc

        chunks: List[SplitChunk] = []
        for idx, doc in enumerate(lc_docs):
            meta: dict[str, Any] = dict(doc.metadata) if doc.metadata else {}
            meta.setdefault("chunk_index", idx)
            chunks.append(
                SplitChunk(
                    content=doc.page_content,
                    index=idx,
                    metadata=meta,
                )
            )

        return chunks


# ── auto-register with factory ─────────────────────────────────────────────
SplitterFactory.register_provider("recursive", RecursiveSplitter)
