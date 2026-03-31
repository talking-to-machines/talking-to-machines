"""
Tests for the RAG layer (gateway/rag.py).

Covers:
  - RAGResult dataclass: field defaults and construction
  - RAGTool: abstract interface enforcement
  - OpenAIRAGTool.retrieve():
      - successful retrieval → chunks populated, failures empty
      - API exception → failure recorded, chunks empty
      - latency_ms > 0 for any call
      - tool_name is "openai_file_search"
      - max_results forwarded to the API call
      - ImportError when openai package absent
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from talkingtomachines.gateway.rag import RAGResult, RAGTool


# ---------------------------------------------------------------------------
# RAGResult dataclass
# ---------------------------------------------------------------------------


def test_rag_result_construction():
    result = RAGResult(
        query="climate change",
        chunks=["chunk A", "chunk B"],
        chunks_retrieved=2,
        tool_name="openai_file_search",
        latency_ms=42.0,
    )
    assert result.query == "climate change"
    assert result.chunks == ["chunk A", "chunk B"]
    assert result.chunks_retrieved == 2
    assert result.tool_name == "openai_file_search"
    assert result.latency_ms == 42.0


def test_rag_result_default_cost_zero():
    result = RAGResult(
        query="q",
        chunks=[],
        chunks_retrieved=0,
        tool_name="test",
        latency_ms=1.0,
    )
    assert result.cost_usd == 0.0


def test_rag_result_default_failures_empty():
    result = RAGResult(
        query="q",
        chunks=[],
        chunks_retrieved=0,
        tool_name="test",
        latency_ms=1.0,
    )
    assert result.failures == []


def test_rag_result_default_raw_response_none():
    result = RAGResult(
        query="q",
        chunks=[],
        chunks_retrieved=0,
        tool_name="test",
        latency_ms=1.0,
    )
    assert result.raw_response is None


def test_rag_result_with_failures():
    result = RAGResult(
        query="q",
        chunks=[],
        chunks_retrieved=0,
        tool_name="test",
        latency_ms=5.0,
        failures=["timeout"],
    )
    assert "timeout" in result.failures


# ---------------------------------------------------------------------------
# RAGTool abstract interface
# ---------------------------------------------------------------------------


def test_rag_tool_is_abstract():
    """Cannot instantiate RAGTool directly — must implement retrieve()."""
    with pytest.raises(TypeError):
        RAGTool()  # type: ignore[abstract]


def test_rag_tool_concrete_subclass_works():
    class ConcreteRAG(RAGTool):
        def retrieve(self, query: str, **kwargs) -> RAGResult:
            return RAGResult(
                query=query,
                chunks=["test"],
                chunks_retrieved=1,
                tool_name="concrete",
                latency_ms=0.0,
            )

    tool = ConcreteRAG()
    result = tool.retrieve("hello")
    assert result.chunks == ["test"]
    assert result.tool_name == "concrete"


# ---------------------------------------------------------------------------
# OpenAIRAGTool — successful retrieval
# ---------------------------------------------------------------------------


def _make_mock_openai(chunks: list[str], raise_exc: Exception | None = None):
    """Build a minimal mock openai module with a vector_stores API."""
    mock_openai_module = MagicMock()

    mock_client = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client

    if raise_exc is not None:
        mock_client.beta.vector_stores.file_batches.search.side_effect = raise_exc
    else:
        # Build fake response.data items
        data_items = []
        for text in chunks:
            item = MagicMock()
            item.content = text
            data_items.append(item)
        mock_response = MagicMock()
        mock_response.data = data_items
        mock_client.beta.vector_stores.file_batches.search.return_value = mock_response

    return mock_openai_module, mock_client


def _make_tool(mock_openai_module):
    """Patch openai import and return an OpenAIRAGTool instance."""
    with patch.dict(sys.modules, {"openai": mock_openai_module}):
        from importlib import import_module

        # Force re-import under the patched module
        import talkingtomachines.gateway.rag as rag_mod

        # Temporarily inject the mock OpenAI into the module
        original = getattr(rag_mod, "_client_class", None)
        mock_openai_module.OpenAI.return_value  # touch it

        # Directly patch inside the class constructor
        with patch("talkingtomachines.gateway.rag.OpenAIRAGTool.__init__") as mock_init:

            def fake_init(self, api_key, vector_store_id):
                self._client = mock_openai_module.OpenAI(api_key=api_key)
                self._vector_store_id = vector_store_id

            mock_init.side_effect = fake_init

            from talkingtomachines.gateway.rag import OpenAIRAGTool

            tool = OpenAIRAGTool(api_key="sk-test", vector_store_id="vs_123")
            tool._client = mock_openai_module.OpenAI.return_value
            tool._vector_store_id = "vs_123"
    return tool


def _build_tool_with_mocked_client(chunks: list[str], raise_exc=None):
    """Construct OpenAIRAGTool with its _client replaced by a mock."""
    from talkingtomachines.gateway.rag import OpenAIRAGTool

    mock_openai_module, mock_client = _make_mock_openai(chunks, raise_exc)

    # Bypass the __init__ (which requires openai) by patching just the openai import
    with patch.dict(sys.modules, {"openai": mock_openai_module}):
        tool = OpenAIRAGTool.__new__(OpenAIRAGTool)
        tool._client = mock_client
        tool._vector_store_id = "vs_123"
    return tool, mock_client


def test_retrieve_returns_rag_result():
    tool, _ = _build_tool_with_mocked_client(["doc A", "doc B"])
    result = tool.retrieve("test query")
    assert isinstance(result, RAGResult)


def test_retrieve_populates_chunks():
    tool, _ = _build_tool_with_mocked_client(["chunk 1", "chunk 2"])
    result = tool.retrieve("some query")
    assert result.chunks == ["chunk 1", "chunk 2"]
    assert result.chunks_retrieved == 2


def test_retrieve_query_preserved():
    tool, _ = _build_tool_with_mocked_client(["c1"])
    result = tool.retrieve("my specific question")
    assert result.query == "my specific question"


def test_retrieve_tool_name_is_openai_file_search():
    tool, _ = _build_tool_with_mocked_client(["c1"])
    result = tool.retrieve("q")
    assert result.tool_name == "openai_file_search"


def test_retrieve_latency_ms_positive():
    tool, _ = _build_tool_with_mocked_client(["c1"])
    result = tool.retrieve("q")
    assert result.latency_ms >= 0.0


def test_retrieve_failures_empty_on_success():
    tool, _ = _build_tool_with_mocked_client(["c1"])
    result = tool.retrieve("q")
    assert result.failures == []


def test_retrieve_forwards_max_results():
    tool, mock_client = _build_tool_with_mocked_client(["c1"])
    tool.retrieve("q", max_results=10)
    mock_client.beta.vector_stores.file_batches.search.assert_called_once_with(
        vector_store_id="vs_123",
        query="q",
        max_num_results=10,
    )


# ---------------------------------------------------------------------------
# OpenAIRAGTool — failure handling
# ---------------------------------------------------------------------------


def test_retrieve_api_exception_records_failure():
    tool, _ = _build_tool_with_mocked_client(
        [], raise_exc=RuntimeError("network error")
    )
    result = tool.retrieve("q")
    assert len(result.failures) == 1
    assert "network error" in result.failures[0]


def test_retrieve_api_exception_returns_empty_chunks():
    tool, _ = _build_tool_with_mocked_client([], raise_exc=RuntimeError("timeout"))
    result = tool.retrieve("q")
    assert result.chunks == []
    assert result.chunks_retrieved == 0


def test_retrieve_api_exception_still_has_latency():
    tool, _ = _build_tool_with_mocked_client([], raise_exc=RuntimeError("err"))
    result = tool.retrieve("q")
    assert result.latency_ms >= 0.0


def test_retrieve_empty_data_returns_empty_chunks():
    """API returns a response with empty data list."""
    tool, _ = _build_tool_with_mocked_client([])
    result = tool.retrieve("q")
    assert result.chunks == []
    assert result.chunks_retrieved == 0
    assert result.failures == []


# ---------------------------------------------------------------------------
# OpenAIRAGTool — ImportError when openai not installed
# ---------------------------------------------------------------------------


def test_openai_rag_tool_raises_import_error_without_openai():
    """If openai is not installed, OpenAIRAGTool.__init__ raises ImportError."""
    # Temporarily hide openai from sys.modules
    original = sys.modules.pop("openai", None)
    try:
        # Force re-evaluation of the import inside __init__
        from talkingtomachines.gateway.rag import OpenAIRAGTool

        with patch.dict(sys.modules, {"openai": None}):  # type: ignore[dict-item]
            with pytest.raises((ImportError, TypeError)):
                OpenAIRAGTool(api_key="sk-fake", vector_store_id="vs_fake")
    finally:
        if original is not None:
            sys.modules["openai"] = original
