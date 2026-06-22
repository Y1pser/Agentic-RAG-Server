"""Unit tests for BaseLLM, Message, ChatResponse, and LLMFactory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory
from rag_mcp_server.src.core.settings import Settings


class MockLLM(BaseLLM):
    """Mock LLM implementation for testing."""

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        return ChatResponse(
            content=f"Echo: {messages[-1].content}",
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset the LLMFactory registry before each test."""
    LLMFactory.clear_registry()
    yield
    LLMFactory.clear_registry()


@pytest.fixture
def settings():
    """Create a minimal Settings instance for testing."""
    return Settings(
        llm={"provider": "mock", "model": "mock-model"},
    )


# ── Message Tests ─────────────────────────────────────────────────────


class TestMessage:
    def test_create_message(self):
        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_message_equality(self):
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="user", content="Hello")
        assert msg1 == msg2

    def test_message_inequality(self):
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="assistant", content="Hello")
        assert msg1 != msg2


# ── ChatResponse Tests ────────────────────────────────────────────────


class TestChatResponse:
    def test_minimal_response(self):
        resp = ChatResponse(content="Answer", model="gpt-4")
        assert resp.content == "Answer"
        assert resp.model == "gpt-4"
        assert resp.usage is None
        assert resp.raw_response is None

    def test_full_response(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        resp = ChatResponse(
            content="Full answer",
            model="gpt-4o",
            usage=usage,
            raw_response={"id": "chatcmpl-123"},
        )
        assert resp.usage == usage
        assert resp.usage["total_tokens"] == 150
        assert resp.raw_response["id"] == "chatcmpl-123"


# ── BaseLLM Tests ─────────────────────────────────────────────────────


class TestBaseLLM:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]

    def test_validate_messages_valid(self):
        llm = MockLLM(Settings())
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        llm.validate_messages(messages)  # Should not raise

    def test_validate_messages_empty_list(self):
        llm = MockLLM(Settings())
        with pytest.raises(ValueError, match="cannot be empty"):
            llm.validate_messages([])

    def test_validate_messages_invalid_type(self):
        llm = MockLLM(Settings())
        with pytest.raises(ValueError, match="is not a Message instance"):
            llm.validate_messages(["not a message"])  # type: ignore[arg-type]

    def test_validate_messages_invalid_role(self):
        llm = MockLLM(Settings())
        with pytest.raises(ValueError, match="invalid role"):
            llm.validate_messages([Message(role="invalid_role", content="test")])

    def test_validate_messages_empty_content(self):
        llm = MockLLM(Settings())
        with pytest.raises(ValueError, match="empty content"):
            llm.validate_messages([Message(role="user", content="   ")])


# ── LLMFactory Tests ──────────────────────────────────────────────────


class TestLLMFactory:
    def test_register_provider(self):
        LLMFactory.register_provider("mock", MockLLM)
        assert LLMFactory.is_registered("mock")

    def test_register_provider_case_insensitive(self):
        LLMFactory.register_provider("MOCK", MockLLM)
        assert LLMFactory.is_registered("mock")
        assert LLMFactory.is_registered("MOCK")

    def test_register_provider_rejects_non_base_llm(self):
        class NotAnLLM:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseLLM"):
            LLMFactory.register_provider("bad", NotAnLLM)

    def test_list_providers_empty(self):
        assert LLMFactory.list_providers() == []

    def test_list_providers_sorted(self):
        LLMFactory.register_provider("openai", MockLLM)
        LLMFactory.register_provider("azure", MockLLM)
        LLMFactory.register_provider("deepseek", MockLLM)
        assert LLMFactory.list_providers() == ["azure", "deepseek", "openai"]

    def test_create_returns_instance(self, settings):
        LLMFactory.register_provider("mock", MockLLM)
        llm = LLMFactory.create(settings)
        assert isinstance(llm, MockLLM)

    def test_create_passes_settings(self, settings):
        LLMFactory.register_provider("mock", MockLLM)
        llm = LLMFactory.create(settings)
        assert llm.settings is settings

    def test_create_passes_override_kwargs(self, settings):
        LLMFactory.register_provider("mock", MockLLM)
        llm = LLMFactory.create(settings, temperature=0.5, extra_param="value")
        assert llm.kwargs == {"temperature": 0.5, "extra_param": "value"}

    def test_create_unknown_provider(self, settings):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMFactory.create(settings)

    def test_create_missing_provider_config(self):
        settings = Settings()  # Empty llm dict
        with pytest.raises(ValueError, match="Missing required configuration"):
            LLMFactory.create(settings)

    def test_is_registered_false(self):
        assert not LLMFactory.is_registered("nonexistent")

    def test_clear_registry(self):
        LLMFactory.register_provider("mock", MockLLM)
        assert LLMFactory.is_registered("mock")
        LLMFactory.clear_registry()
        assert not LLMFactory.is_registered("mock")
        assert LLMFactory.list_providers() == []


# ── MockLLM Integration Test ──────────────────────────────────────────


class TestMockLLM:
    def test_chat_returns_chat_response(self, settings):
        LLMFactory.register_provider("mock", MockLLM)
        llm = LLMFactory.create(settings)
        messages = [Message(role="user", content="Hello")]
        resp = llm.chat(messages)
        assert isinstance(resp, ChatResponse)
        assert resp.content == "Echo: Hello"
        assert resp.model == "mock-model"
        assert resp.usage is not None
        assert resp.usage["total_tokens"] == 15
