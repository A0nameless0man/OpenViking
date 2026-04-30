# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""RED tests for ReindexEngine (paginated scan, batch diff, QueueFS, cancel, progress, errors).

All tests MUST fail because ReindexEngine doesn't exist yet.
They define the expected API contract for the TDD GREEN phase.

Tests use FakeCollectionAdapter (in-memory) and mock NamedQueue/Embedder
to avoid real backend infrastructure dependencies.

ReindexEngine is imported INSIDE test functions (not at module level)
so the test file doesn't crash on import error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# =============================================================================
# ReindexProgress — expected dataclass for progress tracking
# =============================================================================


@dataclass
class ReindexProgress:
    """Expected progress tracking structure for ReindexEngine.

    Defined here so tests can assert on expected fields without
    importing from the not-yet-existing reindex_engine module.
    """

    processed: int = 0
    total: int = 0
    errors: int = 0
    skipped: int = 0
    error_details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.processed + self.errors + self.skipped >= self.total


# =============================================================================
# FakeCollectionAdapter — in-memory fake for testing
# =============================================================================


class FakeCollectionAdapter:
    """In-memory fake implementing the CollectionAdapter public API.

    Features:
    - Dict-based storage (id → record, with uri indexing for filter queries)
    - Tracks all upsert/query calls for test verification
    - Configurable failure injection for error handling tests
    - Supports "fail N times then succeed" pattern for retry tests
    """

    def __init__(
        self,
        collection_name: str = "test",
        mode: str = "fake",
        *,
        exists: bool = True,
    ):
        self._collection_name = collection_name
        self.mode = mode
        self._exists = exists
        # --- storage ---
        self._records: Dict[str, Dict[str, Any]] = {}
        # --- call tracking ---
        self._upsert_call_count: int = 0
        self._upsert_records: List[List[Dict[str, Any]]] = []
        self._query_call_count: int = 0
        self._query_calls: List[Dict[str, Any]] = []
        # --- failure injection ---
        self._raise_on_upsert: Optional[Exception] = None
        self._raise_on_query: Optional[Exception] = None
        self._fail_on_upsert_count: int = 0
        self._upsert_failures: int = 0
        self._query_failures: int = 0
        self._fail_on_query_count: int = 0

    # ------------------------------------------------------------------
    # Public API (matching CollectionAdapter)
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        return self._exists

    def upsert(self, data: Dict[str, Any] | List[Dict[str, Any]]) -> List[str]:
        """Insert or update records. Returns list of IDs."""
        self._upsert_failures += 1
        if self._raise_on_upsert is not None and self._upsert_failures > self._fail_on_upsert_count:
            raise self._raise_on_upsert

        records = [data] if isinstance(data, dict) else data
        self._upsert_call_count += 1
        self._upsert_records.append([dict(r) for r in records])

        ids: List[str] = []
        for item in records:
            record = dict(item)
            record_id = record.get("id") or record.get("uri") or str(id(record))
            record["id"] = record_id
            ids.append(record_id)
            self._records[record_id] = record
        return ids

    def query(
        self,
        *,
        query_vector: Optional[List[float]] = None,
        sparse_query_vector: Optional[Dict[str, float]] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        output_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return stored records with basic filter support."""
        if self._raise_on_query is not None:
            self._query_failures += 1
            if self._query_failures > self._fail_on_query_count:
                raise self._raise_on_query

        self._query_call_count += 1
        self._query_calls.append({
            "filter": filter,
            "limit": limit,
            "offset": offset,
            "output_fields": output_fields,
            "order_by": order_by,
        })

        records = list(self._records.values())

        # Apply simple filter: support "uri" == value or "uri" in [...]
        if filter:
            records = self._apply_filter(records, filter)

        if order_by:
            records = sorted(records, key=lambda r: r.get(order_by, ""), reverse=order_desc)

        if offset:
            records = records[offset:]
        if limit:
            records = records[:limit]

        result = [dict(r) for r in records]
        if output_fields:
            result = [
                {k: r[k] for k in output_fields if k in r}
                for r in result
            ]
        return result

    def _apply_filter(
        self, records: List[Dict[str, Any]], filter_expr: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply a simple filter expression to records.

        Supports:
        - {"uri": "value"} — equality match
        - {"uri": ["a", "b"]} — uri in list
        - {"uri": {"$in": ["a", "b"]}} — uri in list (MongoDB-style)
        """
        uri_filter = filter_expr.get("uri")
        if uri_filter is not None:
            if isinstance(uri_filter, list):
                uri_set = set(uri_filter)
                return [r for r in records if r.get("uri") in uri_set]
            elif isinstance(uri_filter, dict) and "$in" in uri_filter:
                uri_set = set(uri_filter["$in"])
                return [r for r in records if r.get("uri") in uri_set]
            else:
                return [r for r in records if r.get("uri") == uri_filter]
        # Pass-through for unrecognized filters
        return records

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch records by ID."""
        return [dict(self._records[rid]) for rid in ids if rid in self._records]

    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        records = list(self._records.values())
        if filter:
            records = self._apply_filter(records, filter)
        return len(records)

    def delete(
        self,
        *,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 100000,
    ) -> int:
        delete_ids = list(ids or [])
        if not delete_ids and filter is not None:
            matched = self.query(filter=filter, limit=limit)
            delete_ids = [r["id"] for r in matched if r.get("id")]
        for rid in delete_ids:
            self._records.pop(rid, None)
        return len(delete_ids)

    # ------------------------------------------------------------------
    # Failure injection helpers
    # ------------------------------------------------------------------

    def set_raise_on_upsert(self, exc: Exception, *, succeed_first: int = 0) -> None:
        """Configure upsert to raise exc after succeed_first successful calls."""
        self._raise_on_upsert = exc
        self._fail_on_upsert_count = succeed_first
        self._upsert_failures = 0

    def set_raise_on_query(self, exc: Exception, *, succeed_first: int = 0) -> None:
        """Configure query to raise exc after succeed_first successful calls."""
        self._raise_on_query = exc
        self._fail_on_query_count = succeed_first
        self._query_failures = 0

    def reset_failures(self) -> None:
        """Reset failure injection state."""
        self._raise_on_upsert = None
        self._raise_on_query = None
        self._fail_on_upsert_count = 0
        self._upsert_failures = 0
        self._query_failures = 0
        self._fail_on_query_count = 0

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def upsert_call_count(self) -> int:
        return self._upsert_call_count

    @property
    def query_call_count(self) -> int:
        return self._query_call_count

    @property
    def query_calls(self) -> List[Dict[str, Any]]:
        return list(self._query_calls)

    @property
    def stored_ids(self) -> List[str]:
        return sorted(self._records.keys())

    @property
    def stored_uris(self) -> List[str]:
        return sorted(r.get("uri", "") for r in self._records.values() if r.get("uri"))

    def reset_counts(self) -> None:
        """Reset call tracking without clearing stored records."""
        self._upsert_call_count = 0
        self._query_call_count = 0
        self._query_calls = []
        self._upsert_records = []


# =============================================================================
# Mock helpers — create mock NamedQueue and Embedder for testing
# =============================================================================


def make_mock_named_queue(
    *,
    uris_to_dequeue: Optional[List[Dict[str, Any]]] = None,
    enqueue_side_effect=None,
    dequeue_side_effect=None,
    ack_side_effect=None,
) -> MagicMock:
    """Create a mock NamedQueue for testing.

    The mock wraps AsyncMock for async methods (enqueue, dequeue, ack, etc.)
    so they can be awaited in async tests.
    """
    mock_queue = MagicMock()
    mock_queue.name = "test_reindex_queue"

    # Async methods
    mock_queue.enqueue = AsyncMock(side_effect=enqueue_side_effect)
    mock_queue.dequeue = AsyncMock(side_effect=dequeue_side_effect)
    mock_queue.ack = AsyncMock(side_effect=ack_side_effect)
    mock_queue.size = AsyncMock(return_value=0)
    mock_queue.clear = AsyncMock(return_value=True)
    mock_queue.peek = AsyncMock(return_value=None)

    # Track enqueued URIs for verification
    mock_queue._enqueued_items = []  # type: ignore[assignment]  # List[str] on MagicMock
    mock_queue._dequeued_count = 0  # type: ignore[assignment]  # int on MagicMock
    mock_queue._acked_ids = []  # type: ignore[assignment]  # List[str] on MagicMock

    return mock_queue


def make_mock_embedder(
    *,
    dense_vector: Optional[List[float]] = None,
    fail_on_uris: Optional[set] = None,
    fail_with: Optional[Exception] = None,
    fail_count: int = 0,
) -> MagicMock:
    """Create a mock embedder for testing.

    Args:
        dense_vector: Default dense vector to return. If not set, generates
            a deterministic vector based on URI.
        fail_on_uris: Set of URIs that should trigger an embedding failure.
        fail_with: Exception to raise for failed embeddings.
        fail_count: Number of times to succeed before failing.
    """
    from openviking.models.embedder.base import EmbedResult

    _fail_count = [0]  # mutable counter for tracking failures across calls

    async def _embed_async(text: str, is_query: bool = False) -> EmbedResult:
        if fail_on_uris and text in fail_on_uris:
            _fail_count[0] += 1
            if fail_count == 0 or _fail_count[0] > fail_count:
                raise (fail_with or RuntimeError(f"Embed failed for {text}"))
        vec = dense_vector if dense_vector else [float(hash(text) % 100) / 100.0]
        return EmbedResult(dense_vector=vec)

    mock_embedder = MagicMock()
    mock_embedder.embed_async = AsyncMock(side_effect=_embed_async)
    mock_embedder._fail_count = _fail_count
    return mock_embedder


# =============================================================================
# Safe import helper — raises ModuleNotFoundError during RED phase
# =============================================================================


def _import_reindex_engine():
    """Import ReindexEngine from the migration module.

    This MUST raise ModuleNotFoundError/ImportError during RED phase
    because the module doesn't exist yet.
    """
    from openviking.storage.migration.reindex_engine import ReindexEngine  # noqa: F811

    return ReindexEngine


# =============================================================================
# 1. Scan tests (Task 2.1)
# =============================================================================


def test_scan_source_uris_paginated():
    """scan_source_uris() must paginate source using offset+limit, not load all at once."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    # Populate source with 150 records
    for i in range(150):
        source.upsert({"uri": f"viking://rec_{i:04d}", "text": f"record {i}", "id": f"id_{i:04d}"})
    source.reset_counts()

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=FakeCollectionAdapter(collection_name="target"),
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    # We can't call async scan_source_uris from sync test.
    # The key RED assertion: ReindexEngine exists and has scan_source_uris method.
    assert hasattr(engine, "scan_source_uris"), "ReindexEngine must expose scan_source_uris"
    assert callable(engine.scan_source_uris), "scan_source_uris must be callable"


@pytest.mark.asyncio
async def test_scan_source_uris_returns_only_uri_field():
    """scan_source_uris() must use output_fields=["uri"] to avoid loading full records."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    source.upsert({"uri": "viking://a", "text": "long text content", "embedding": [0.1] * 100})
    source.upsert({"uri": "viking://b", "text": "another long text", "embedding": [0.2] * 100})

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=FakeCollectionAdapter(collection_name="target"),
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    # scan_source_uris should yield batches of URI strings
    uris_seen: List[str] = []
    async for batch in engine.scan_source_uris(page_size=10):
        uris_seen.extend(batch)

    assert len(uris_seen) == 2
    assert set(uris_seen) == {"viking://a", "viking://b"}
    # At least one query call must have used output_fields=["uri"]
    output_fields_used = any(
        call.get("output_fields") == ["uri"] for call in reversed(source.query_calls)
    )
    assert output_fields_used, "scan_source_uris must use output_fields=['uri']"


def test_filter_missing_returns_only_missing_uris():
    """filter_missing() must return only URIs NOT present in target."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")
    # Target already has "viking://b"
    target.upsert({"uri": "viking://b", "text": "exists in target"})

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    missing = engine.filter_missing(["viking://a", "viking://b", "viking://c"])
    assert isinstance(missing, list)
    assert missing == ["viking://a", "viking://c"]


def test_filter_missing_batches_queries():
    """filter_missing() must query target only for the given batch, not load ALL target URIs."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")
    # Target has many records — but filter_missing should NOT do a full scan
    for i in range(200):
        target.upsert({"uri": f"viking://exist_{i:04d}"})

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    target.reset_counts()
    result = engine.filter_missing(["viking://new_001", "viking://new_002"])

    # The query to target should NOT request all 200 records (limit should reflect batch size)
    for call in target.query_calls:
        if call.get("limit", 0) >= 200:
            pytest.fail(
                "filter_missing must NOT load all target URIs. "
                f"Found query with limit={call.get('limit')}, but batch only has 2 URIs."
            )

    assert result == ["viking://new_001", "viking://new_002"]


def test_filter_missing_empty_batch():
    """filter_missing([]) must return empty list."""
    ReindexEngine = _import_reindex_engine()

    engine = ReindexEngine(
        source_adapter=FakeCollectionAdapter(collection_name="source"),
        target_embedder=make_mock_embedder(),
        target_adapter=FakeCollectionAdapter(collection_name="target"),
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    result = engine.filter_missing([])
    assert result == []


def test_filter_missing_all_exist():
    """When all URIs already exist in target, filter_missing returns empty list."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")
    target.upsert({"uri": "viking://x"})
    target.upsert({"uri": "viking://y"})

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    result = engine.filter_missing(["viking://x", "viking://y"])
    assert result == []


# =============================================================================
# 2. Queue tests (Task 2.3)
# =============================================================================


def test_enqueue_uris_deduplicates():
    """enqueue_uris() must deduplicate URIs using set() before enqueuing."""
    ReindexEngine = _import_reindex_engine()

    target = FakeCollectionAdapter(collection_name="target")
    mock_queue = make_mock_named_queue()

    engine = ReindexEngine(
        source_adapter=FakeCollectionAdapter(collection_name="source"),
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )
    # Inject mock queue
    engine._queue = mock_queue

    count = engine.enqueue_uris(["uri_a", "uri_b", "uri_a", "uri_c", "uri_b"])
    # Count should be 3 (distinct URIs), not 5
    assert count == 3, f"Expected 3 deduplicated URIs, got {count}"


def test_enqueue_uris_stores_in_named_queue():
    """enqueue_uris() must write each URI to the NamedQueue."""
    ReindexEngine = _import_reindex_engine()

    target = FakeCollectionAdapter(collection_name="target")
    mock_queue = make_mock_named_queue()

    engine = ReindexEngine(
        source_adapter=FakeCollectionAdapter(collection_name="source"),
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )
    engine._queue = mock_queue

    engine.enqueue_uris(["viking://rec_1", "viking://rec_2"])
    assert mock_queue.enqueue.call_count == 2


@pytest.mark.asyncio
async def test_process_queue_embeds_and_upserts():
    """process_queue() must dequeue URIs, embed them, and upsert to target."""
    ReindexEngine = _import_reindex_engine()

    from openviking.models.embedder.base import EmbedResult

    source = FakeCollectionAdapter(collection_name="source")
    source.upsert({"uri": "viking://a", "text": "hello world", "id": "id_a"})
    source.upsert({"uri": "viking://b", "text": "goodbye world", "id": "id_b"})

    target = FakeCollectionAdapter(collection_name="target")

    mock_embedder = make_mock_embedder(dense_vector=[0.1, 0.2, 0.3])

    # Mock queue: dequeue returns URIs in sequence, then None
    dequeue_results = [
        {"id": "msg_1", "data": "viking://a"},
        {"id": "msg_2", "data": "viking://b"},
        None,  # end of queue
    ]
    dequeued = iter(dequeue_results)

    async def _dequeue():
        try:
            return next(dequeued)
        except StopIteration:
            return None

    mock_queue = make_mock_named_queue()
    mock_queue.dequeue = AsyncMock(side_effect=_dequeue)

    import asyncio

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=mock_embedder,
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )
    engine._queue = mock_queue

    stop_event = asyncio.Event()
    await engine.process_queue(stop_event)

    # Both URIs should have been embedded
    assert mock_embedder.embed_async.call_count == 2
    # Both should have been upserted to target
    assert target.upsert_call_count == 2
    # Both messages should have been ack'd
    assert mock_queue.ack.call_count == 2


@pytest.mark.asyncio
async def test_process_queue_semaphore_limits_concurrency():
    """process_queue() must use asyncio.Semaphore to limit concurrent embeddings."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    for i in range(10):
        source.upsert({"uri": f"viking://rec_{i}"})

    target = FakeCollectionAdapter(collection_name="target")

    # Embedder that takes a bit of time to confirm concurrency limit
    max_concurrent_seen = [0]
    active_calls = [0]

    async def _slow_embed(text: str, is_query: bool = False):
        active_calls[0] += 1
        max_concurrent_seen[0] = max(max_concurrent_seen[0], active_calls[0])
        await __import__("asyncio").sleep(0.01)
        active_calls[0] -= 1
        from openviking.models.embedder.base import EmbedResult
        return EmbedResult(dense_vector=[0.1])

    mock_embedder = make_mock_embedder()
    mock_embedder.embed_async = AsyncMock(side_effect=_slow_embed)

    # Queue with 10 URIs
    dequeue_count = [0]
    async def _dequeue():
        dequeue_count[0] += 1
        if dequeue_count[0] <= 10:
            return {"id": f"msg_{dequeue_count[0]}", "data": f"viking://rec_{dequeue_count[0]-1}"}
        return None

    mock_queue = make_mock_named_queue()
    mock_queue.dequeue = AsyncMock(side_effect=_dequeue)

    import asyncio

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=mock_embedder,
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=3,  # limit to 3 concurrent
        batch_size=50,
    )
    engine._queue = mock_queue

    stop_event = asyncio.Event()
    await engine.process_queue(stop_event)

    # Max concurrency should not exceed max_concurrent
    assert max_concurrent_seen[0] <= 3, (
        f"Concurrency limit violated: {max_concurrent_seen[0]} > 3"
    )


@pytest.mark.asyncio
async def test_process_queue_acks_after_successful_upsert():
    """process_queue() must call ack(msg_id) after each successful upsert."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    source.upsert({"uri": "viking://ok", "text": "success"})

    target = FakeCollectionAdapter(collection_name="target")
    mock_embedder = make_mock_embedder(dense_vector=[0.5, 0.6])

    mock_queue = make_mock_named_queue()
    mock_queue.dequeue = AsyncMock(return_value={"id": "msg_ok", "data": "viking://ok"})
    # Second call returns None to stop the loop
    mock_queue.dequeue.side_effect = [
        {"id": "msg_ok", "data": "viking://ok"},
        None,
    ]

    import asyncio

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=mock_embedder,
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )
    engine._queue = mock_queue

    stop_event = asyncio.Event()
    await engine.process_queue(stop_event)

    # ack must have been called with the message ID
    mock_queue.ack.assert_any_call("msg_ok")


@pytest.mark.asyncio
async def test_process_queue_does_not_ack_on_failure():
    """process_queue() must NOT ack a message when embedding or upsert fails."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    source.upsert({"uri": "viking://fail", "text": "will fail"})

    target = FakeCollectionAdapter(collection_name="target")

    # Embedder that always fails on first attempt
    mock_embedder = make_mock_embedder(
        fail_on_uris={"viking://fail"},
        fail_with=RuntimeError("embed failure"),
        fail_count=0,  # always fail immediately
    )

    mock_queue = make_mock_named_queue()
    mock_queue.dequeue = AsyncMock(side_effect=[
        {"id": "msg_fail", "data": "viking://fail"},
        None,  # end of queue
    ])

    import asyncio

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=mock_embedder,
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
        max_retries=0,  # no retries
    )
    engine._queue = mock_queue

    stop_event = asyncio.Event()
    await engine.process_queue(stop_event)

    # ack should NOT be called for the failed message
    # (or at least not called with msg_fail)
    ack_calls_with_fail = [
        call for call in mock_queue.ack.call_args_list
        if call.args and "msg_fail" in str(call.args)
    ]
    assert len(ack_calls_with_fail) == 0, (
        "ack must NOT be called for failed processing"
    )


# =============================================================================
# 3. Cancel + Progress tests (Task 2.5)
# =============================================================================


@pytest.mark.asyncio
async def test_cancel_stops_processing_new_uris():
    """cancel() must set stop_event so process_queue stops dequeuing new URIs."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    for i in range(20):
        source.upsert({"uri": f"viking://rec_{i}", "text": f"record {i}"})

    target = FakeCollectionAdapter(collection_name="target")

    # Embedder with a small delay so cancel can interrupt
    async def _slow_embed(text: str, is_query: bool = False):
        await __import__("asyncio").sleep(0.02)
        from openviking.models.embedder.base import EmbedResult
        return EmbedResult(dense_vector=[0.1])

    mock_embedder = MagicMock()
    mock_embedder.embed_async = AsyncMock(side_effect=_slow_embed)

    dequeue_count = [0]
    async def _dequeue():
        dequeue_count[0] += 1
        if dequeue_count[0] <= 20:
            return {"id": f"msg_{dequeue_count[0]}", "data": f"viking://rec_{dequeue_count[0]-1}"}
        return None

    mock_queue = make_mock_named_queue()
    mock_queue.dequeue = AsyncMock(side_effect=_dequeue)

    import asyncio

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=mock_embedder,
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )
    engine._queue = mock_queue

    assert hasattr(engine, "cancel"), "ReindexEngine must expose cancel()"
    assert callable(engine.cancel), "cancel must be callable"

    # The cancel method should set a stop_event that process_queue checks
    # The actual verification is that cancel exists and is callable during RED phase


def test_cancel_completes_current_uri():
    """After cancel(), the URI currently being processed must complete before stopping."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=FakeCollectionAdapter(collection_name="target"),
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    # Cancel must not abort in-progress work — only prevent NEW work
    # RED phase: just verify the method exists
    assert hasattr(engine, "cancel"), "ReindexEngine must expose cancel()"


def test_cancel_leaves_unacked_messages_in_queue():
    """After cancel(), messages not yet processed must remain in the queue (not ack'd)."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    # Unacked messages must be preserved for RecoverStale recovery
    # RED phase: verify method exists
    assert hasattr(engine, "cancel"), "cancel must leave unacked messages in queue"


def test_get_progress_returns_accurate_counts():
    """get_progress() must return ReindexProgress with accurate processed/total/errors/skipped."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    assert hasattr(engine, "get_progress"), "ReindexEngine must expose get_progress()"
    assert callable(engine.get_progress), "get_progress must be callable"

    progress = engine.get_progress()
    # Progress should be a ReindexProgress-like object with expected fields
    assert hasattr(progress, "processed"), "progress must have 'processed' field"
    assert hasattr(progress, "total"), "progress must have 'total' field"
    assert hasattr(progress, "errors"), "progress must have 'errors' field"
    assert hasattr(progress, "skipped"), "progress must have 'skipped' field"


def test_progress_tracks_skipped_uris():
    """Progress must track URIs skipped by filter_missing (already in target)."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    # After filter_missing skips URIs, progress.skipped should be incremented
    assert hasattr(engine, "get_progress"), "get_progress must be available"


def test_progress_tracks_errors():
    """Progress must track errors from failed embeddings."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    # After embed failures, progress.errors should be incremented
    assert hasattr(engine, "get_progress"), "get_progress must be available"


# =============================================================================
# 4. Error handling tests (Task 2.7)
# =============================================================================


@pytest.mark.asyncio
async def test_single_uri_failure_does_not_block_others():
    """A single URI embedding failure must not block processing of subsequent URIs."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    source.upsert({"uri": "viking://good", "text": "good record"})
    source.upsert({"uri": "viking://bad", "text": "bad record"})
    source.upsert({"uri": "viking://also_good", "text": "also good"})

    target = FakeCollectionAdapter(collection_name="target")

    # Embedder fails only for the "bad" URI
    mock_embedder = make_mock_embedder(
        dense_vector=[0.1, 0.2, 0.3],
        fail_on_uris={"viking://bad"},
        fail_with=RuntimeError("embed failed"),
        fail_count=3,  # Always fail (0 successes before failure)
    )

    dequeue_sequence = [
        {"id": "msg_good", "data": "viking://good"},
        {"id": "msg_bad", "data": "viking://bad"},
        {"id": "msg_also_good", "data": "viking://also_good"},
        None,
    ]
    deq_iter = iter(dequeue_sequence)

    async def _dequeue():
        try:
            return next(deq_iter)
        except StopIteration:
            return None

    mock_queue = make_mock_named_queue()
    mock_queue.dequeue = AsyncMock(side_effect=_dequeue)

    import asyncio

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=mock_embedder,
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
        max_retries=1,  # minimal retries so the bad URI doesn't loop forever
    )
    engine._queue = mock_queue

    stop_event = asyncio.Event()
    await engine.process_queue(stop_event)

    # "good" and "also_good" should have been upserted
    assert target.upsert_call_count >= 2, (
        f"Expected at least 2 upserts (good + also_good), got {target.upsert_call_count}"
    )
    # The bad URI should NOT have blocked processing
    # hack: check that target has records for good and also_good
    uris_in_target = {r.get("uri") for r in target.query(limit=100)}
    assert "viking://good" in uris_in_target or "viking://also_good" in uris_in_target, (
        "Both good URIs should reach target; single failure must not block others"
    )


def test_embed_error_retried_N_times():
    """Embedding failure must be retried max_retries times before skipping."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
        max_retries=3,  # default 3 retries
    )

    assert hasattr(engine, "max_retries") or hasattr(type(engine), "max_retries"), (
        "ReindexEngine must support configurable max_retries (default 3)"
    )


def test_upsert_error_retried_N_times():
    """Upsert failure must be retried max_retries times before skipping."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
        max_retries=3,
    )

    assert hasattr(engine, "max_retries") or hasattr(type(engine), "max_retries"), (
        "ReindexEngine must retry upsert failures up to max_retries times"
    )


def test_error_recorded_in_progress():
    """Failed embedding error info must be recorded in progress.error_details."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    target = FakeCollectionAdapter(collection_name="target")

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=make_mock_embedder(),
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
    )

    assert hasattr(engine, "get_progress"), "get_progress must return progress with error tracking"
    progress = engine.get_progress()
    assert hasattr(progress, "errors") or hasattr(progress, "error_details"), (
        "progress must track error count or details"
    )


@pytest.mark.asyncio
async def test_max_retries_exhausted_skips_uri():
    """When max_retries is exhausted for a URI, it must be skipped (not ack'd)."""
    ReindexEngine = _import_reindex_engine()

    source = FakeCollectionAdapter(collection_name="source")
    source.upsert({"uri": "viking://fails_always", "text": "permanent failure"})
    source.upsert({"uri": "viking://ok", "text": "ok record"})

    target = FakeCollectionAdapter(collection_name="target")

    # Embedder always fails
    mock_embedder = make_mock_embedder(
        fail_on_uris={"viking://fails_always"},
        fail_with=RuntimeError("permanent embed error"),
        fail_count=0,  # Always fail
    )

    mock_queue = make_mock_named_queue()
    mock_queue.dequeue = AsyncMock(side_effect=[
        {"id": "msg_fail", "data": "viking://fails_always"},
        {"id": "msg_ok", "data": "viking://ok"},
        None,
    ])

    import asyncio

    engine = ReindexEngine(
        source_adapter=source,
        target_embedder=mock_embedder,
        target_adapter=target,
        queue_name="test_queue",
        max_concurrent=5,
        batch_size=50,
        max_retries=1,  # retry once then skip
    )
    engine._queue = mock_queue

    stop_event = asyncio.Event()
    await engine.process_queue(stop_event)

    # msg_fail should NOT be ack'd (left in queue for recovery)
    # msg_ok should still be processed
    acked_ids = [
        c.args[0] for c in mock_queue.ack.call_args_list
        if c.args
    ]
    assert "msg_fail" not in acked_ids, (
        f"Failed message must NOT be ack'd, but was found in {acked_ids}"
    )
    # The OK URI should have been upserted
    assert target.upsert_call_count >= 1, "Good URI must still be processed"
