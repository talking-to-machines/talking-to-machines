"""Retrieval-Augmented Generation (RAG) support.

Provides an abstract :class:`RAGTool` interface and a concrete
:class:`OpenAIRAGTool` implementation backed by the OpenAI File Search
(vector store) API. The abstract interface is designed for future
extension to other retrieval backends.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Normalised result from a RAG retrieval call.

    Attributes:
        query: The search query that was submitted.
        chunks: List of retrieved text chunks.
        chunks_retrieved: Number of chunks successfully retrieved.
        tool_name: Identifier of the RAG tool that produced this result.
        latency_ms: Wall-clock time for the retrieval call in milliseconds.
        cost_usd: Estimated cost in US dollars. Defaults to ``0.0``.
        failures: List of error messages if any failures occurred.
        raw_response: The unprocessed SDK response object. Defaults to ``None``.
    """

    query: str
    chunks: list[str]
    chunks_retrieved: int
    tool_name: str
    latency_ms: float
    cost_usd: float = 0.0
    failures: list[str] = field(default_factory=list)
    raw_response: Any = None


class RAGTool(ABC):
    """Abstract interface for all RAG implementations.

    Subclasses must implement :meth:`retrieve` to search a document
    store and return matching text chunks.
    """

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> RAGResult:
        """Retrieve relevant text chunks for *query*.

        Args:
            query: The natural-language search query.
            **kwargs: Provider-specific retrieval options.

        Returns:
            A :class:`RAGResult` containing the retrieved chunks and
            associated metadata.
        """


class OpenAIRAGTool(RAGTool):
    """RAG implementation backed by the OpenAI File Search (vector store) API.

    Args:
        api_key: OpenAI API key.
        vector_store_id: The identifier of the OpenAI vector store to
            search against.

    Raises:
        ImportError: If the ``openai`` package is not installed.
    """

    def __init__(self, api_key: str, vector_store_id: str):
        """Initialise the OpenAI RAG tool with an API key and vector store."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required for RAG") from exc

        self._client = OpenAI(api_key=api_key)
        self._vector_store_id = vector_store_id

    def retrieve(self, query: str, max_results: int = 5, **kwargs) -> RAGResult:
        """Search the vector store for chunks relevant to *query*.

        Args:
            query: The natural-language search query.
            max_results: Maximum number of chunks to retrieve. Defaults
                to ``5``.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A :class:`RAGResult` with the retrieved text chunks and
            timing metadata. Any API errors are captured in the
            ``failures`` list rather than raised.
        """
        t0 = time.perf_counter()
        failures: list[str] = []
        chunks: list[str] = []
        raw = None

        try:
            response = self._client.beta.vector_stores.file_batches.search(
                vector_store_id=self._vector_store_id,
                query=query,
                max_num_results=max_results,
            )
            raw = response
            for result in getattr(response, "data", []):
                text = getattr(result, "content", "") or ""
                if text:
                    chunks.append(text)
        except Exception as exc:
            failures.append(str(exc))
            logger.warning("RAG retrieval failed: %s", exc)

        latency_ms = (time.perf_counter() - t0) * 1000
        return RAGResult(
            query=query,
            chunks=chunks,
            chunks_retrieved=len(chunks),
            tool_name="openai_file_search",
            latency_ms=latency_ms,
            failures=failures,
            raw_response=raw,
        )
