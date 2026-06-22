"""Tests for vector store factory routing and BaseVectorStore contract.

Covers B1.4: VectorStore abstract interface and factory.
"""

import pytest
from rag_mcp_server.src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
)
from rag_mcp_server.src.libs.vector_store.vector_store_factory import VectorStoreFactory
from rag_mcp_server.src.core.settings import Settings


# ---------------------------------------------------------------------------
# Fake / stub implementations for testing
# ---------------------------------------------------------------------------

class FakeVectorStore(BaseVectorStore):
    """In-memory fake vector store for testing."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self._records: dict[str, VectorRecord] = {}
        self._counter = 0

    def add(self, records, trace=None, **kwargs):
        ids = []
        for rec in records:
            self._counter += 1
            rid = rec.id or f"vec-{self._counter}"
            self._records[rid] = VectorRecord(
                id=rid,
                embedding=rec.embedding,
                metadata=rec.metadata,
                text=rec.text,
            )
            ids.append(rid)
        return ids

    def get(self, ids, trace=None, **kwargs):
        return [self._records[rid] for rid in ids if rid in self._records]

    def query(self, embedding, top_k=5, trace=None, **kwargs):
        # Brute-force cosine similarity for testing
        scored = []
        for rid, rec in self._records.items():
            sim = _cosine_similarity(embedding, rec.embedding)
            scored.append(QueryResult(id=rid, score=sim, metadata=rec.metadata))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def delete(self, ids, trace=None, **kwargs):
        removed = 0
        for rid in ids:
            if rid in self._records:
                del self._records[rid]
                removed += 1
        return removed

    def count(self) -> int:
        """Return number of stored records (non-standard helper for testing)."""
        return len(self._records)


class AnotherFakeVectorStore(BaseVectorStore):
    """A different fake for provider switching verification."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings

    def add(self, records, trace=None, **kwargs):
        return [r.id or f"a-{i}" for i, r in enumerate(records)]

    def get(self, ids, trace=None, **kwargs):
        return []

    def query(self, embedding, top_k=5, trace=None, **kwargs):
        return [QueryResult(id="fake", score=1.0, metadata={})]

    def delete(self, ids, trace=None, **kwargs):
        return len(ids)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# VectorRecord & QueryResult tests
# ---------------------------------------------------------------------------

class TestVectorRecord:
    """Tests for VectorRecord dataclass."""

    def test_defaults(self):
        """ID should default to None; metadata defaults to empty dict."""
        rec = VectorRecord(embedding=[0.1, 0.2])
        assert rec.id is None
        assert rec.metadata == {}
        assert rec.text is None
        assert rec.embedding == [0.1, 0.2]

    def test_full_record(self):
        """All fields should be set correctly."""
        rec = VectorRecord(
            id="doc-1",
            embedding=[1.0, 0.0],
            metadata={"source": "test.pdf", "page": 1},
            text="sample content",
        )
        assert rec.id == "doc-1"
        assert rec.metadata["source"] == "test.pdf"
        assert rec.text == "sample content"


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_default_metadata(self):
        """Metadata should default to empty dict."""
        qr = QueryResult(id="id-1", score=0.95)
        assert qr.metadata == {}

    def test_fields(self):
        """All fields accessible."""
        qr = QueryResult(id="chunk-3", score=0.82, metadata={"page": 5})
        assert qr.id == "chunk-3"
        assert qr.score == 0.82
        assert qr.metadata["page"] == 5


# ---------------------------------------------------------------------------
# BaseVectorStore contract tests
# ---------------------------------------------------------------------------

class TestBaseVectorStore:
    """Verify BaseVectorStore enforces its contract."""

    def test_cannot_instantiate_abstract(self):
        """BaseVectorStore should not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseVectorStore()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """A concrete subclass should be instantiable and functional."""
        store = FakeVectorStore()
        assert isinstance(store, BaseVectorStore)
        assert store.count() == 0

    def test_add_and_get_roundtrip(self):
        """Adding records then getting them should return the same data."""
        store = FakeVectorStore()
        rec = VectorRecord(embedding=[1.0, 0.0, 0.0], metadata={"x": 1})
        ids = store.add([rec])
        assert len(ids) == 1
        assert ids[0].startswith("vec-")

        retrieved = store.get(ids)
        assert len(retrieved) == 1
        assert retrieved[0].embedding == [1.0, 0.0, 0.0]
        assert retrieved[0].metadata == {"x": 1}

    def test_query_returns_sorted_by_score(self):
        """Query should return results sorted by descending similarity."""
        store = FakeVectorStore()
        store.add([VectorRecord(embedding=[1.0, 0.0], metadata={"label": "A"})])
        store.add([VectorRecord(embedding=[0.0, 1.0], metadata={"label": "B"})])

        results = store.query(embedding=[1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert results[0].metadata["label"] == "A"  # most similar

    def test_query_respects_top_k(self):
        """Query should honor top_k limit."""
        store = FakeVectorStore()
        for i in range(10):
            store.add([VectorRecord(embedding=[float(i), 0.0])])

        results = store.query(embedding=[5.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_delete_removes_records(self):
        """Delete should remove records and return count."""
        store = FakeVectorStore()
        ids = store.add([
            VectorRecord(embedding=[1.0, 0.0]),
            VectorRecord(embedding=[0.0, 1.0]),
        ])

        removed = store.delete([ids[0]])
        assert removed == 1
        assert store.count() == 1
        assert store.get([ids[0]]) == []
        assert len(store.get([ids[1]])) == 1

    def test_get_nonexistent_returns_empty(self):
        """Getting non-existent IDs should return empty list."""
        store = FakeVectorStore()
        assert store.get(["ghost"]) == []


# ---------------------------------------------------------------------------
# VectorStoreFactory tests
# ---------------------------------------------------------------------------

class TestVectorStoreFactory:
    """Tests for vector store factory: registration, creation, error handling."""

    def setup_method(self):
        """Ensure clean registry."""
        VectorStoreFactory.clear_registry()

    def teardown_method(self):
        """Clean up."""
        VectorStoreFactory.clear_registry()

    # -- Registration -------------------------------------------------------

    def test_register_provider_adds_to_registry(self):
        """register_provider should register a name."""
        VectorStoreFactory.register_provider("fake", FakeVectorStore)
        assert "fake" in VectorStoreFactory._PROVIDERS
        assert VectorStoreFactory.is_registered("fake")

    def test_register_non_subclass_raises(self):
        """Non-BaseVectorStore class should raise."""

        class NotAVectorStore:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseVectorStore"):
            VectorStoreFactory.register_provider("bad", NotAVectorStore)

    def test_register_is_case_insensitive(self):
        """Names should be lower-cased."""
        VectorStoreFactory.register_provider("FAKE", FakeVectorStore)
        assert VectorStoreFactory.is_registered("fake")

    # -- Creation -----------------------------------------------------------

    def test_create_returns_store_instance(self):
        """Factory.create should instantiate registered provider."""
        VectorStoreFactory.register_provider("fake", FakeVectorStore)
        settings = Settings()
        settings.vector_store = {"backend": "fake"}

        store = VectorStoreFactory.create(settings)
        assert isinstance(store, BaseVectorStore)
        assert isinstance(store, FakeVectorStore)

    def test_create_passes_settings_to_provider(self):
        """Provider constructor should receive settings."""
        VectorStoreFactory.register_provider("fake", FakeVectorStore)
        settings = Settings()
        settings.vector_store = {"backend": "fake", "persist_directory": "/tmp/db"}

        store = VectorStoreFactory.create(settings)
        rec = VectorRecord(embedding=[1.0, 0.0])
        ids = store.add([rec])
        assert len(ids) == 1

    def test_create_with_override_kwargs(self):
        """Override kwargs should forward to constructor."""
        VectorStoreFactory.register_provider("fake", FakeVectorStore)
        settings = Settings()
        settings.vector_store = {"backend": "fake"}

        store = VectorStoreFactory.create(settings, collection="my_collection")
        assert isinstance(store, FakeVectorStore)

    # -- Error handling -----------------------------------------------------

    def test_missing_backend_config_raises(self):
        """Missing vector_store config should raise."""
        settings = Settings()
        with pytest.raises(ValueError, match="vector_store.backend"):
            VectorStoreFactory.create(settings)

    def test_unknown_provider_raises(self):
        """Unregistered provider should raise."""
        settings = Settings()
        settings.vector_store = {"backend": "nonexistent"}
        with pytest.raises(ValueError, match="nonexistent"):
            VectorStoreFactory.create(settings)

    def test_instantiation_failure_wraps_error(self):
        """Init failures should wrap in RuntimeError."""

        class BrokenStore(BaseVectorStore):
            def __init__(self, settings=None, **kwargs):
                raise RuntimeError("Disk full")

            def add(self, records, trace=None, **kwargs):
                return []
            def get(self, ids, trace=None, **kwargs):
                return []
            def query(self, embedding, top_k=5, trace=None, **kwargs):
                return []
            def delete(self, ids, trace=None, **kwargs):
                return 0

        VectorStoreFactory.register_provider("broken", BrokenStore)
        settings = Settings()
        settings.vector_store = {"backend": "broken"}

        with pytest.raises(RuntimeError, match="Failed to instantiate"):
            VectorStoreFactory.create(settings)

    # -- Listing & querying ------------------------------------------------

    def test_list_providers_returns_sorted_names(self):
        """list_providers should return sorted names."""
        VectorStoreFactory.register_provider("chroma", FakeVectorStore)
        VectorStoreFactory.register_provider("annoy", AnotherFakeVectorStore)
        assert VectorStoreFactory.list_providers() == ["annoy", "chroma"]

    def test_is_registered_returns_false_for_unknown(self):
        """Should return False for unregistered."""
        assert not VectorStoreFactory.is_registered("pinecone")

    def test_clear_registry_removes_all(self):
        """clear_registry should empty registry."""
        VectorStoreFactory.register_provider("fake", FakeVectorStore)
        VectorStoreFactory.clear_registry()
        assert VectorStoreFactory.list_providers() == []


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------

class TestVectorStoreSettingsIntegration:
    """Integration with Settings."""

    def test_settings_has_vector_store_field(self):
        """Settings should expose vector_store config section."""
        settings = Settings()
        assert hasattr(settings, "vector_store")
        assert isinstance(settings.vector_store, dict)
