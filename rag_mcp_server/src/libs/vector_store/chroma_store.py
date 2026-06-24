"""ChromaDB vector store implementation.

Provides the default vector store backend using ChromaDB — an open-source
embeddings database that runs locally with zero operational overhead.

Design Principles Applied:
- Pluggable: Implements BaseVectorStore, auto-registers with factory.
- Config-Driven: Reads persist_directory / collection_name from settings.
- Self-Contained: Uses ChromaDB PersistentClient for local storage; falls
  back to in-memory mode when no persist_directory is configured (testing).
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    QueryResult,
    VectorRecord,
)
from rag_mcp_server.src.libs.vector_store.vector_store_factory import (
    VectorStoreFactory,
)


def _short_uuid() -> str:
    """Generate a short unique identifier (8 hex chars)."""
    return uuid.uuid4().hex[:8]


class ChromaStore(BaseVectorStore):
    """ChromaDB-backed vector store.

    Stores embeddings, metadata, and document text in a ChromaDB collection.
    Supports both persistent (on-disk) and ephemeral (in-memory) modes.

    Configuration (from ``settings.vector_store``):

    .. code-block:: yaml

        vector_store:
          backend: "chroma"
          persist_directory: "data/chroma"
          collection_name: "knowledge_hub"

    Extra keyword arguments passed to the constructor override config values.
    """

    # Default collection name if none specified in config
    DEFAULT_COLLECTION_NAME: str = "knowledge_hub"
    # Default persist directory
    DEFAULT_PERSIST_DIR: str = "data/chroma"

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the ChromaDB vector store.

        Args:
            settings: Application settings containing vector_store config.
            **kwargs: Overrides — recognised keys: ``persist_directory``,
                ``collection_name``, ``client`` (pre-built client for testing).
        """
        import chromadb

        vs_cfg: dict[str, Any] = getattr(settings, "vector_store", None) or {}

        # ── resolve persist_directory ────────────────────────────────────
        self.persist_directory: str = kwargs.pop(
            "persist_directory", None
        ) or vs_cfg.get("persist_directory", self.DEFAULT_PERSIST_DIR)

        # ── resolve collection_name ──────────────────────────────────────
        self.collection_name: str = kwargs.pop(
            "collection_name", None
        ) or vs_cfg.get("collection_name", self.DEFAULT_COLLECTION_NAME)

        # ── allow injecting a pre-built client (testing) ─────────────────
        prebuilt_client = kwargs.pop("client", None)
        if prebuilt_client is not None:
            self._client = prebuilt_client
        elif self.persist_directory:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self._client = chromadb.Client()

        # ── get-or-create the collection ─────────────────────────────────
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Store remaining kwargs for introspection / debugging
        self._kwargs: dict[str, Any] = kwargs

    # ── public interface ─────────────────────────────────────────────────────

    def add(
        self,
        records: List[VectorRecord],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add vector records to the ChromaDB collection.

        Args:
            records: List of VectorRecord objects. Records without an ``id``
                will be assigned an auto-generated one.
            trace: Optional TraceContext (reserved for Phase B-5).
            **kwargs: Passed through to ChromaDB ``collection.add()``.

        Returns:
            List of record IDs in the same order as the input records.

        Raises:
            ValueError: If *records* is empty.
            RuntimeError: If the ChromaDB operation fails.
        """
        if not records:
            raise ValueError("records must not be empty")

        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[dict[str, Any]] = []
        documents: List[str] = []

        for i, rec in enumerate(records):
            rid = rec.id or f"chunk-{_short_uuid()}"
            ids.append(rid)
            embeddings.append(list(rec.embedding))
            # ChromaDB requires non-empty metadata dicts — inject a sentinel
            # key when none is provided so the call doesn't fail.
            meta = dict(rec.metadata) if rec.metadata else None
            if meta is not None and len(meta) == 0:
                meta = {"_chunk_index": i}
            elif meta is None:
                meta = {"_chunk_index": i}
            metadatas.append(meta)
            documents.append(rec.text or "")

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ChromaStore.add failed for {len(records)} record(s): {exc}"
            ) from exc

        return ids

    def get(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[VectorRecord]:
        """Retrieve vector records by their IDs.

        Args:
            ids: List of record identifiers to fetch.
            trace: Optional TraceContext (reserved for Phase B-5).
            **kwargs: Passed through to ChromaDB ``collection.get()``.

        Returns:
            List of VectorRecord objects for IDs that exist (missing IDs
            are silently skipped, matching ChromaDB behaviour).

        Raises:
            RuntimeError: If the ChromaDB operation fails.
        """
        if not ids:
            return []

        try:
            result = self._collection.get(
                ids=ids,
                include=["embeddings", "metadatas", "documents"],
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ChromaStore.get failed for {len(ids)} id(s): {exc}"
            ) from exc

        records: List[VectorRecord] = []
        # Use explicit None-checks because ChromaDB may return numpy arrays
        # whose truth value is ambiguous (raises ValueError with `or []`).
        result_ids_raw = result.get("ids")
        result_ids: List[str] = list(result_ids_raw) if result_ids_raw is not None else []
        result_embeddings_raw = result.get("embeddings")
        result_embeddings = list(result_embeddings_raw) if result_embeddings_raw is not None else []
        result_metadatas_raw = result.get("metadatas")
        result_metadatas = list(result_metadatas_raw) if result_metadatas_raw is not None else []
        result_documents_raw = result.get("documents")
        result_documents = list(result_documents_raw) if result_documents_raw is not None else []

        for i, rid in enumerate(result_ids):
            records.append(
                VectorRecord(
                    id=rid,
                    embedding=result_embeddings[i] if i < len(result_embeddings) else [],
                    metadata=result_metadatas[i] if i < len(result_metadatas) else {},
                    text=result_documents[i] if i < len(result_documents) else None,
                )
            )

        return records

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[QueryResult]:
        """Query the store for records most similar to a given embedding.

        Uses cosine distance (configured at collection creation) and converts
        distances to similarity scores via ``score = 1 - distance``.

        Args:
            embedding: The query vector to compare against stored records.
            top_k: Maximum number of results to return.
            trace: Optional TraceContext (reserved for Phase B-5).
            **kwargs: Passed through to ChromaDB ``collection.query()``.

        Returns:
            List of QueryResult objects sorted by descending similarity score.

        Raises:
            ValueError: If *embedding* is empty.
            RuntimeError: If the ChromaDB query fails.
        """
        self.validate_embedding(embedding)

        # Pop filters if provided (ChromaDB where-clause)
        where_filter = kwargs.pop("where", None)

        try:
            result = self._collection.query(
                query_embeddings=[list(embedding)],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "distances"],
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ChromaStore.query failed: {exc}"
            ) from exc

        query_results: List[QueryResult] = []

        # ChromaDB returns nested lists: ids[0], distances[0], metadatas[0].
        # Use explicit None-checks to avoid numpy array truthiness issues.
        result_ids_raw = result.get("ids")
        result_ids: List[str] = (
            list(result_ids_raw[0]) if result_ids_raw is not None and len(result_ids_raw) > 0 else []
        )
        result_distances_raw = result.get("distances")
        result_distances: List[float] = (
            list(result_distances_raw[0]) if result_distances_raw is not None and len(result_distances_raw) > 0 else []
        )
        result_metadatas_raw = result.get("metadatas")
        result_metadatas: List[dict] = (
            list(result_metadatas_raw[0]) if result_metadatas_raw is not None and len(result_metadatas_raw) > 0 else []
        )

        for i, rid in enumerate(result_ids):
            distance = result_distances[i] if i < len(result_distances) else 0.0
            # Convert cosine distance [0, 2] to similarity score [-1, 1]
            score = 1.0 - distance
            meta = result_metadatas[i] if i < len(result_metadatas) else {}
            query_results.append(
                QueryResult(id=rid, score=score, metadata=dict(meta))
            )

        return query_results

    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> int:
        """Delete vector records by their IDs.

        Counts existing records before deletion to return an accurate count.

        Args:
            ids: List of record identifiers to delete.
            trace: Optional TraceContext (reserved for Phase B-5).
            **kwargs: Passed through to ChromaDB ``collection.delete()``.

        Returns:
            Number of records actually removed.

        Raises:
            RuntimeError: If the ChromaDB delete operation fails.
        """
        if not ids:
            return 0

        # Count existing records before deleting
        try:
            before = self._collection.get(ids=ids)
            before_ids = before.get("ids")
            count_before = len(list(before_ids)) if before_ids is not None else 0
        except Exception as exc:
            raise RuntimeError(
                f"ChromaStore.delete failed during pre-count: {exc}"
            ) from exc

        if count_before == 0:
            return 0

        try:
            self._collection.delete(ids=ids, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"ChromaStore.delete failed: {exc}"
            ) from exc

        return count_before

    # ── helpers ──────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of records in the collection.

        Non-standard convenience method for testing and introspection.
        """
        return self._collection.count()


# ── auto-register with factory ─────────────────────────────────────────────
VectorStoreFactory.register_provider("chroma", ChromaStore)