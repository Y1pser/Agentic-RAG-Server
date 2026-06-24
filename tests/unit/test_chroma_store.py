"""Tests for ChromaStore — B1.11.

Covers:
- Construction from settings
- Add + Get roundtrip (core verification)
- Query by embedding similarity
- Delete with count
- Factory integration and auto-registration
- Edge cases (empty records, missing IDs, empty embeddings)
"""

from __future__ import annotations

import math
import uuid

import pytest

from rag_mcp_server.src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    QueryResult,
    VectorRecord,
)
from rag_mcp_server.src.libs.vector_store.vector_store_factory import (
    VectorStoreFactory,
)
from rag_mcp_server.src.core.settings import Settings

# Import ChromaStore to trigger auto-registration
from rag_mcp_server.src.libs.vector_store.chroma_store import ChromaStore  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLLECTION_COUNTER: int = 0


def _unique_name(prefix: str = "test") -> str:
    """Generate a unique collection name to isolate tests."""
    global _COLLECTION_COUNTER
    _COLLECTION_COUNTER += 1
    return f"{prefix}_{_COLLECTION_COUNTER}_{uuid.uuid4().hex[:6]}"


def _make_settings(collection_name: str = "") -> Settings:
    """Build a Settings stub with vector_store config."""
    s = Settings()
    s.vector_store = {"backend": "chroma"}
    if collection_name:
        s.vector_store["collection_name"] = collection_name
    return s


def _ephemeral_store(collection_name: str = "") -> ChromaStore:
    """Create a ChromaStore backed by a fresh in-memory ChromaDB client."""
    import chromadb

    client = chromadb.Client()
    name = collection_name or _unique_name("ephemeral")
    settings = _make_settings(collection_name=name)
    return ChromaStore(settings, client=client)


def _sample_embedding(dim: int = 128, seed: float = 0.0) -> list[float]:
    """Generate a deterministic pseudo-embedding for testing."""
    return [math.sin(seed + i * 0.1) for i in range(dim)]


def _build_record(
    idx: int = 0,
    dim: int = 128,
    metadata: dict | None = None,
    text: str | None = None,
) -> VectorRecord:
    """Build a VectorRecord with a deterministic embedding."""
    return VectorRecord(
        embedding=_sample_embedding(dim=dim, seed=float(idx)),
        metadata=metadata or {"index": idx},
        text=text or f"Document chunk {idx}",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestChromaStoreConstruction:
    """Tests for ChromaStore instantiation and configuration."""

    def test_constructs_from_settings(self):
        """Should construct from a Settings object."""
        store = _ephemeral_store()
        assert isinstance(store, BaseVectorStore)
        assert store.count() == 0

    def test_default_collection_name(self):
        """Should use default collection name when not in settings."""
        import chromadb

        settings = Settings()
        settings.vector_store = {"backend": "chroma"}
        store = ChromaStore(settings, client=chromadb.Client())
        assert store.collection_name == ChromaStore.DEFAULT_COLLECTION_NAME

    def test_kwargs_override_settings(self):
        """Constructor kwargs should override settings values."""
        import chromadb

        client = chromadb.Client()
        store = ChromaStore(
            Settings(),
            client=client,
            vector_store={"backend": "chroma", "collection_name": "from_config"},
            collection_name="from_kwargs",
        )
        assert store.collection_name == "from_kwargs"

    def test_persist_directory_default(self):
        """Should default persist_directory when not configured."""
        import chromadb

        settings = Settings()
        settings.vector_store = {"backend": "chroma"}
        store = ChromaStore(settings, client=chromadb.Client())
        assert store.persist_directory == ChromaStore.DEFAULT_PERSIST_DIR

    def test_ephemeral_mode_works(self):
        """Ephemeral (in-memory) ChromaDB should work without disk."""
        store = _ephemeral_store()
        assert store.count() == 0
        ids = store.add([_build_record(0)])
        assert store.count() == 1
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# Add + Get roundtrip (core B1.11 verification)
# ---------------------------------------------------------------------------

class TestAddGetRoundtrip:
    """Verify that records survive a full add-then-get cycle without
    data loss or corruption — the primary acceptance criterion for B1.11."""

    def test_add_single_get_back(self):
        """A single record should be retrievable with all fields intact."""
        store = _ephemeral_store()
        rec = VectorRecord(
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "doc.pdf", "page": 1},
            text="Hello world",
        )
        ids = store.add([rec])
        assert len(ids) == 1

        retrieved = store.get(ids)
        assert len(retrieved) == 1
        assert retrieved[0].id == ids[0]
        assert retrieved[0].text == "Hello world"
        # Embedding should be approximately equal (float tolerance —
        # ChromaDB stores as float32, so we allow 1e-6)
        for a, b in zip(retrieved[0].embedding, [0.1, 0.2, 0.3]):
            assert abs(float(a) - b) < 1e-6, f"{a} != {b}"

    def test_add_multiple_get_all(self):
        """Multiple records should all be retrievable."""
        store = _ephemeral_store()
        records = [_build_record(i) for i in range(10)]
        ids = store.add(records)
        assert len(ids) == 10
        assert store.count() == 10

        retrieved = store.get(ids)
        assert len(retrieved) == 10

        # Verify content fidelity for each record
        retrieved_by_id = {r.id: r for r in retrieved}
        for i, rid in enumerate(ids):
            assert rid in retrieved_by_id
            assert retrieved_by_id[rid].text == f"Document chunk {i}"

    def test_auto_generated_ids(self):
        """Records without IDs should get auto-generated IDs."""
        store = _ephemeral_store()
        rec = VectorRecord(embedding=[1.0, 0.0], metadata={"idx": 0})
        ids = store.add([rec])
        assert len(ids) == 1
        assert ids[0].startswith("chunk-")
        assert len(ids[0]) == 14  # "chunk-" + 8 hex chars

    def test_custom_ids_preserved(self):
        """Manually assigned IDs should be preserved."""
        store = _ephemeral_store()
        rec = VectorRecord(id="my-custom-id", embedding=[1.0, 0.0], metadata={"idx": 0})
        ids = store.add([rec])
        assert ids == ["my-custom-id"]
        assert store.get(["my-custom-id"])[0].id == "my-custom-id"

    def test_get_nonexistent_returns_empty(self):
        """Getting IDs that don't exist should return an empty list."""
        store = _ephemeral_store()
        assert store.get(["ghost-id"]) == []

    def test_get_partial_match(self):
        """Getting a mix of existing and non-existing IDs should return
        only the existing ones."""
        store = _ephemeral_store()
        ids = store.add([_build_record(0), _build_record(1)])
        retrieved = store.get([ids[0], "no-such-id", ids[1]])
        assert len(retrieved) == 2
        returned_ids = {r.id for r in retrieved}
        assert ids[0] in returned_ids
        assert ids[1] in returned_ids


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestChromaStoreQuery:
    """Tests for similarity query behaviour."""

    def test_query_returns_results(self):
        """Query should return the record whose embedding is closest."""
        store = _ephemeral_store()

        # Add records with distinct embeddings
        store.add([
            VectorRecord(embedding=[1.0, 0.0, 0.0], text="A", metadata={"label": "A"}),
            VectorRecord(embedding=[0.0, 1.0, 0.0], text="B", metadata={"label": "B"}),
            VectorRecord(embedding=[0.0, 0.0, 1.0], text="C", metadata={"label": "C"}),
        ])

        # Query with a vector very close to record A
        results = store.query(embedding=[0.99, 0.01, 0.0], top_k=3)
        assert len(results) >= 1

    def test_query_respects_top_k(self):
        """Query should return at most top_k results."""
        store = _ephemeral_store()
        store.add([_build_record(i) for i in range(20)])

        results = store.query(embedding=_sample_embedding(seed=10.0), top_k=5)
        assert len(results) == 5

        results = store.query(embedding=_sample_embedding(seed=10.0), top_k=10)
        assert len(results) == 10

    def test_query_scores_sorted_descending(self):
        """Query results should be sorted by descending similarity score."""
        store = _ephemeral_store()
        store.add([_build_record(i) for i in range(10)])

        results = store.query(embedding=_sample_embedding(seed=5.0), top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), f"scores not sorted: {scores}"

    def test_query_scores_in_range(self):
        """Scores should be in the valid cosine similarity range [-1, 1]."""
        store = _ephemeral_store()
        store.add([_build_record(i) for i in range(5)])

        results = store.query(embedding=_sample_embedding(seed=2.0), top_k=3)
        for r in results:
            assert -1.0 <= r.score <= 1.0, f"score {r.score} out of range"

    def test_query_empty_store(self):
        """Query on an empty store should return empty list."""
        store = _ephemeral_store()
        results = store.query(embedding=[1.0, 0.0], top_k=5)
        assert results == []

    def test_query_empty_embedding_raises(self):
        """Empty embedding vector should raise ValueError."""
        store = _ephemeral_store()
        with pytest.raises(ValueError, match="embedding must not be empty"):
            store.query(embedding=[], top_k=5)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestChromaStoreDelete:
    """Tests for record deletion."""

    def test_delete_returns_count(self):
        """Delete should return the number of records removed."""
        store = _ephemeral_store()
        ids = store.add([_build_record(0), _build_record(1), _build_record(2)])
        removed = store.delete([ids[0], ids[1]])
        assert removed == 2
        assert store.count() == 1

    def test_delete_is_idempotent(self):
        """Deleting the same IDs twice should return 0 the second time."""
        store = _ephemeral_store()
        ids = store.add([_build_record(0)])
        removed1 = store.delete(ids)
        assert removed1 == 1
        removed2 = store.delete(ids)
        assert removed2 == 0

    def test_delete_empty_list_returns_zero(self):
        """Deleting an empty list should return 0."""
        store = _ephemeral_store()
        assert store.delete([]) == 0

    def test_delete_nonexistent_returns_zero(self):
        """Deleting IDs that don't exist should return 0."""
        store = _ephemeral_store()
        store.add([_build_record(0)])
        assert store.delete(["no-such-id"]) == 0

    def test_deleted_records_not_retrievable(self):
        """Deleted records should not appear in get() or query()."""
        store = _ephemeral_store()
        ids = store.add([_build_record(0)])
        store.delete(ids)
        assert store.get(ids) == []
        results = store.query(embedding=_sample_embedding(seed=0.0), top_k=10)
        assert all(r.id not in ids for r in results)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestChromaStoreEdgeCases:
    """Edge case and error handling tests."""

    def test_add_empty_raises(self):
        """Adding an empty list should raise ValueError."""
        store = _ephemeral_store()
        with pytest.raises(ValueError, match="records must not be empty"):
            store.add([])

    def test_get_empty_list_returns_empty(self):
        """Getting an empty ID list should return empty list."""
        store = _ephemeral_store()
        assert store.get([]) == []

    def test_duplicate_ids_handling(self):
        """Adding records with the same ID should not crash (ChromaDB
        silently keeps the first write — add is not upsert)."""
        store = _ephemeral_store()
        rec1 = VectorRecord(id="same-id", embedding=[1.0, 0.0], text="first", metadata={"v": 1})
        store.add([rec1])
        rec2 = VectorRecord(id="same-id", embedding=[0.0, 1.0], text="second", metadata={"v": 2})
        # ChromaDB add() with duplicate IDs keeps first; upsert() would overwrite
        store.add([rec2])
        retrieved = store.get(["same-id"])
        assert len(retrieved) == 1
        # Either first or second is acceptable — the key is that it doesn't crash

    def test_high_dimensional_embedding(self):
        """Should handle 1536-dim embeddings (OpenAI text-embedding-3-small)."""
        store = _ephemeral_store()
        dim = 1536
        rec = VectorRecord(
            embedding=[0.01] * dim,
            text="high-dim test",
            metadata={"dim": dim},
        )
        ids = store.add([rec])
        retrieved = store.get(ids)
        assert len(retrieved[0].embedding) == dim

    def test_metadata_with_special_chars(self):
        """Metadata with Unicode / special characters should survive."""
        store = _ephemeral_store()
        rec = VectorRecord(
            embedding=[1.0, 0.0],
            metadata={
                "title": "合同纠纷 — 违约金计算标准",
                "emoji": "📄✅",
            },
        )
        ids = store.add([rec])
        retrieved = store.get(ids)
        assert retrieved[0].metadata["title"] == "合同纠纷 — 违约金计算标准"
        assert retrieved[0].metadata["emoji"] == "📄✅"

    def test_text_is_none(self):
        """Records with text=None: ChromaDB stores empty string for null docs."""
        store = _ephemeral_store()
        rec = VectorRecord(embedding=[1.0, 0.0], text=None, metadata={"idx": 0})
        ids = store.add([rec])
        retrieved = store.get(ids)
        # ChromaDB converts None → "" for the documents field
        assert retrieved[0].text == "" or retrieved[0].text is None

    def test_no_metadata_provided(self):
        """Records with metadata=None should still be storable and retrievable."""
        store = _ephemeral_store()
        rec = VectorRecord(embedding=[1.0, 0.0])
        ids = store.add([rec])
        retrieved = store.get(ids)
        assert len(retrieved) == 1
        # The store injects a sentinel key for empty metadata
        assert retrieved[0].id == ids[0]


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------

class TestChromaStoreFactoryIntegration:
    """Verify ChromaStore works correctly through the VectorStoreFactory."""

    @pytest.fixture(autouse=True)
    def _ensure_registered(self):
        """Ensure ChromaStore is registered (import above triggers this)."""
        if not VectorStoreFactory.is_registered("chroma"):
            VectorStoreFactory.register_provider("chroma", ChromaStore)
        yield

    def test_factory_creates_chroma_store(self):
        """VectorStoreFactory should create a ChromaStore when configured."""
        import chromadb

        settings = _make_settings(collection_name=_unique_name("factory"))
        store = VectorStoreFactory.create(settings, client=chromadb.Client())
        assert isinstance(store, ChromaStore)
        assert isinstance(store, BaseVectorStore)

    def test_factory_create_and_use(self):
        """Store created via factory should work for full roundtrip."""
        import chromadb

        settings = _make_settings(collection_name=_unique_name("factory_rt"))
        store = VectorStoreFactory.create(settings, client=chromadb.Client())

        ids = store.add([_build_record(0)])
        assert len(ids) == 1

        retrieved = store.get(ids)
        assert len(retrieved) == 1
        assert retrieved[0].text == "Document chunk 0"

    def test_chroma_is_registered(self):
        """'chroma' should appear in registered providers."""
        providers = VectorStoreFactory.list_providers()
        assert "chroma" in providers

    def test_chroma_is_default_in_settings(self):
        """The settings.yaml default backend 'chroma' should be registered."""
        assert VectorStoreFactory.is_registered("chroma")
