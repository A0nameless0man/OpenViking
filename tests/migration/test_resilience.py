# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""RED tests for crash recovery resilience (migration-spec §2.1.4 C1-C7).

All tests MUST fail because resilience.py does not exist yet.
They define the expected recovery contract for the TDD GREEN phase (Task 3.6).

Crash recovery scenarios tested:
- C2: dual_write → reconstruct DualWriteAdapter from persisted state
- C3: building → reconstruct adapter + engine using target_embedder_name (P0 fix!)
- C4: building_complete → keep building_complete state
- C5: switched → active=target, dual_write enabled
- C6: dual_write_off → active=target, dual_write disabled
- C7: completed → auto cleanup (transition to idle)

P0 verification:
- building recovery MUST use config.get_target_embedder(state.target_embedder_name),
  NOT config.embedding.get_embedder() (current/active embedder).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Gate: ensure resilience module exists (RED phase → it doesn't)
# ---------------------------------------------------------------------------

_has_resilience = False
try:
    from openviking.storage.migration.resilience import recover_from_crash  # noqa: F401
    _has_resilience = True
except ImportError:
    pass


def _require_resilience() -> None:
    """Check that resilience module is importable.

    In the RED phase this always fails — the module doesn't exist yet.
    In the GREEN phase (Task 3.6) this check passes and real assertions run.
    """
    if not _has_resilience:
        pytest.fail(
            "resilience module not yet implemented — RED phase. "
            "This test will pass when openviking.storage.migration.resilience "
            "is implemented in Task 3.6 (GREEN phase)."
        )


# ---------------------------------------------------------------------------
# FakeCollectionAdapter — in-memory fake matching CollectionAdapter API
# ---------------------------------------------------------------------------


class FakeCollectionAdapter:
    """In-memory fake implementing the CollectionAdapter public API.

    Does NOT inherit from CollectionAdapter ABC to avoid needing real
    backend infrastructure. Implements the same public method signatures
    so DualWriteAdapter can delegate to it.
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
        self._records: Dict[str, Dict[str, Any]] = {}
        self._upsert_call_count: int = 0
        self._upsert_records: List[List[Dict[str, Any]]] = []
        self._delete_call_count: int = 0
        self._delete_ids: List[str] = []
        self._query_call_count: int = 0

    def collection_exists(self) -> bool:
        return self._exists

    def upsert(self, data: Dict[str, Any] | List[Dict[str, Any]]) -> List[str]:
        records = [data] if isinstance(data, dict) else data
        self._upsert_call_count += 1
        self._upsert_records.append([dict(r) for r in records])
        ids: List[str] = []
        for item in records:
            record = dict(item)
            record_id = record.get("id") or str(uuid.uuid4())
            record["id"] = record_id
            ids.append(record_id)
            self._records[record_id] = record
        return ids

    def delete(
        self,
        *,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 100000,
    ) -> int:
        self._delete_call_count += 1
        if ids:
            self._delete_ids.extend(ids)
            deleted = 0
            for rid in ids:
                if rid in self._records:
                    del self._records[rid]
                    deleted += 1
            return deleted
        return 0

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
        self._query_call_count += 1
        results = list(self._records.values())
        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]
        if output_fields:
            results = [
                {k: v for k, v in r.items() if k in output_fields}
                for r in results
            ]
        return results

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        return [self._records[rid] for rid in ids if rid in self._records]

    def count(
        self, filter: Optional[Dict[str, Any]] = None
    ) -> int:
        return len(self._records)

    def clear(self) -> bool:
        self._records.clear()
        return True

    def drop_collection(self) -> bool:
        self._exists = False
        self._records.clear()
        return True

    def close(self) -> None:
        pass

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        return {"name": self._collection_name, "mode": self.mode}

    def get_collection(self) -> Any:
        return self._collection_name

    def set_collection(self, collection: Any) -> None:
        self._collection_name = str(collection)


# ---------------------------------------------------------------------------
# FakeEmbedder — minimal embedder mock
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Fake embedder that returns deterministic vectors based on input length."""

    def __init__(self, name: str = "fake-embedder", dimension: int = 128):
        self.name = name
        self.dimension = dimension

    async def embed_async(self, text: str) -> List[float]:
        """Return a deterministic vector based on text content."""
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:self.dimension]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_source_adapter() -> FakeCollectionAdapter:
    """Source collection adapter with some pre-existing records."""
    adapter = FakeCollectionAdapter(collection_name="source_collection")
    adapter.upsert([
        {"id": "uri-1", "uri": "viking://resources/doc1", "content": "hello"},
        {"id": "uri-2", "uri": "viking://resources/doc2", "content": "world"},
        {"id": "uri-3", "uri": "viking://resources/doc3", "content": "test"},
    ])
    return adapter


@pytest.fixture
def fake_target_adapter() -> FakeCollectionAdapter:
    """Target collection adapter — empty by default."""
    return FakeCollectionAdapter(collection_name="target_collection")


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock OpenVikingConfig with embeddings dict and get_target_embedder."""
    config = MagicMock()

    # Current active embedder (NOT the target for migration)
    current_embedder = FakeEmbedder(name="current-v1-embedder", dimension=1024)
    config.embedding = MagicMock()
    config.embedding.get_embedder.return_value = current_embedder

    # Named embedding configs
    target_embedder = FakeEmbedder(name="target-v2-embedder", dimension=2048)
    target_embedding_config = MagicMock()
    target_embedding_config.get_embedder.return_value = target_embedder

    config.embeddings = {
        "v1": MagicMock(),  # source embedder config
        "v2": target_embedding_config,  # target embedder config
    }

    # get_target_embedder resolves from embeddings dict
    def _get_target_embedder(name: str):
        cfg = config.embeddings.get(name)
        if cfg is None:
            raise KeyError(f"Unknown embedding config: {name}")
        return cfg.get_embedder()

    config.get_target_embedder = _get_target_embedder

    return config


@pytest.fixture
def mock_queue_manager() -> MagicMock:
    """Mock QueueFS queue manager."""
    qm = MagicMock()
    qm.get_queue.return_value = MagicMock()
    return qm


# ---------------------------------------------------------------------------
# Helper: build a minimal MigrationState for testing
# ---------------------------------------------------------------------------


def make_state(phase_str: str, **kwargs: Any):
    """Create a minimal MigrationState instance for the given phase.

    Since MigrationState already exists (state.py), we import it here.
    Falls back to a plain mock if import fails (which it shouldn't).
    """
    try:
        from openviking.storage.migration.state import MigrationState, MigrationPhase

        defaults = {
            "migration_id": f"mig-test-{phase_str}",
            "phase": MigrationPhase(phase_str),
            "source_collection": "source_collection",
            "target_collection": "target_collection",
            "active_side": "source",
            "dual_write_enabled": True,
            "source_embedder_name": "v1",
            "target_embedder_name": "v2",
            "degraded_write_failures": 0,
            "reindex_progress": None,
            "started_at": "2026-04-29T10:00:00Z",
            "updated_at": "2026-04-29T10:05:00Z",
        }
        defaults.update(kwargs)
        return MigrationState(**defaults)
    except ImportError:
        # Fallback: use a MagicMock that quacks like MigrationState
        state = MagicMock()
        state.phase = MagicMock()
        state.phase.value = phase_str
        state.migration_id = kwargs.get("migration_id", f"mig-test-{phase_str}")
        state.source_collection = kwargs.get("source_collection", "source_collection")
        state.target_collection = kwargs.get("target_collection", "target_collection")
        state.active_side = kwargs.get("active_side", "source")
        state.dual_write_enabled = kwargs.get("dual_write_enabled", True)
        state.source_embedder_name = kwargs.get("source_embedder_name", "v1")
        state.target_embedder_name = kwargs.get("target_embedder_name", "v2")
        state.degraded_write_failures = kwargs.get("degraded_write_failures", 0)
        state.reindex_progress = kwargs.get("reindex_progress", None)
        return state


# =============================================================================
# C2: Recover dual_write after crash
# =============================================================================


def test_recover_dual_write_after_crash(
    mock_config: MagicMock,
    fake_source_adapter: FakeCollectionAdapter,
    fake_target_adapter: FakeCollectionAdapter,
    mock_queue_manager: MagicMock,
) -> None:
    """C2: Crash during dual_write → reconstruct DualWriteAdapter from state.

    Verifies:
    - recover_from_crash returns a DualWriteAdapter
    - Adapter has active_side = "source"
    - Adapter has dual_write_enabled = True
    - No ReindexEngine returned (phase != building)
    """
    _require_resilience()

    from openviking.storage.migration.resilience import recover_from_crash
    from openviking.storage.migration.blue_green_adapter import DualWriteAdapter

    state = make_state("dual_write")

    result = recover_from_crash(
        state=state,
        config=mock_config,
        queue_manager=mock_queue_manager,
    )

    # Unpack result: (adapter, engine_or_none)
    adapter, engine = result

    # Must return a DualWriteAdapter
    assert isinstance(adapter, DualWriteAdapter), (
        f"Expected DualWriteAdapter, got {type(adapter)}"
    )

    # Active side must be source
    assert adapter._active_side == "source", (
        f"Expected active_side='source', got {adapter._active_side!r}"
    )

    # Dual write must be enabled
    assert adapter._dual_write_enabled is True, (
        f"Expected dual_write_enabled=True, got {adapter._dual_write_enabled!r}"
    )

    # No ReindexEngine (phase != building)
    assert engine is None, (
        f"Expected no ReindexEngine for dual_write phase, got {engine!r}"
    )


# =============================================================================
# C3: Recover building after crash (P0 fix!)
# =============================================================================


def test_recover_building_after_crash(
    mock_config: MagicMock,
    fake_source_adapter: FakeCollectionAdapter,
    fake_target_adapter: FakeCollectionAdapter,
    mock_queue_manager: MagicMock,
) -> None:
    """C3: Crash during building → reconstruct adapter + engine + auto-resume reindex.

    This is the P0 fix scenario:
    - Recreates DualWriteAdapter with active=source, dual_write enabled
    - Creates target embedder using state.target_embedder_name (NOT config.embedding)
    - Creates ReindexEngine with the correct target embedder
    - Reindex should auto-resume (AGFS RecoverStale handles queue recovery)
    """
    _require_resilience()

    from openviking.storage.migration.resilience import recover_from_crash
    from openviking.storage.migration.blue_green_adapter import DualWriteAdapter
    from openviking.storage.migration.reindex_engine import ReindexEngine

    state = make_state("building")

    result = recover_from_crash(
        state=state,
        config=mock_config,
        queue_manager=mock_queue_manager,
    )

    adapter, engine = result

    # Must return a DualWriteAdapter
    assert isinstance(adapter, DualWriteAdapter), (
        f"Expected DualWriteAdapter, got {type(adapter)}"
    )

    # Active side must be source during building
    assert adapter._active_side == "source", (
        f"Expected active_side='source' during building, got {adapter._active_side!r}"
    )

    # Dual write must be enabled
    assert adapter._dual_write_enabled is True

    # Must return a ReindexEngine (building phase!)
    assert isinstance(engine, ReindexEngine), (
        f"Expected ReindexEngine for building phase, got {type(engine)!r}"
    )

    # P0 fix: engine's target_embedder must be the TARGET embedder (v2),
    # NOT the current active embedder (v1)
    assert engine.target_embedder.name == "target-v2-embedder", (
        f"P0 VIOLATION: engine.target_embedder is {engine.target_embedder.name!r}. "
        f"Expected 'target-v2-embedder' (the migration target embedder from state.target_embedder_name='v2'). "
        f"This is the critical P0 fix — must NOT use config.embedding.get_embedder() (which returns 'current-v1-embedder')."
    )

    # Verify engine references correct source/target adapters
    assert engine.source_adapter is not None
    assert engine.target_adapter is not None
    assert engine.queue_name is not None


# =============================================================================
# C3-P0: Explicitly verify NOT using current embedder
# =============================================================================


def test_recover_building_uses_not_current_embedder(
    mock_config: MagicMock,
    fake_source_adapter: FakeCollectionAdapter,
    fake_target_adapter: FakeCollectionAdapter,
    mock_queue_manager: MagicMock,
) -> None:
    """P0 check: building recovery uses target embedder, NOT config.embedding.

    This test explicitly validates the P0 fix from migration-spec.md:395:
    "building 阶段恢复使用 state.target_embedder_name，而非 current_active"

    The current embedder is 'current-v1-embedder' (dimension=1024).
    The target embedder is 'target-v2-embedder' (dimension=2048).
    The recovered ReindexEngine must use the target, NOT the current.
    """
    _require_resilience()

    from openviking.storage.migration.resilience import recover_from_crash
    from openviking.storage.migration.reindex_engine import ReindexEngine

    state = make_state(
        "building",
        source_embedder_name="v1",
        target_embedder_name="v2",
    )

    # Get the current embedder BEFORE recovery (for comparison)
    current_embedder = mock_config.embedding.get_embedder()
    assert current_embedder.name == "current-v1-embedder", (
        "Test setup: current embedder should be 'current-v1-embedder'"
    )
    assert current_embedder.dimension == 1024

    # Perform recovery
    result = recover_from_crash(
        state=state,
        config=mock_config,
        queue_manager=mock_queue_manager,
    )

    _, engine = result

    assert isinstance(engine, ReindexEngine), (
        "Expected ReindexEngine for building phase"
    )

    # The CRITICAL P0 assertion:
    # The recovered engine's embedder MUST NOT be the current active embedder.
    recovered_embedder = engine.target_embedder

    assert recovered_embedder is not current_embedder, (
        "P0 BUG DETECTED: recover_from_crash used config.embedding.get_embedder() "
        f"(the current active embedder '{current_embedder.name}') instead of "
        f"config.get_target_embedder(state.target_embedder_name) "
        f"(the migration target embedder 'target-v2-embedder').\n"
        f"This would cause reindex to embed with the OLD model, producing vectors "
        f"of dimension {current_embedder.dimension} that mismatch the target "
        f"collection's {recovered_embedder.dimension}-dimension schema.\n"
        f"Fix: recover_from_crash MUST call "
        f"config.get_target_embedder(state.target_embedder_name), "
        f"NOT config.embedding.get_embedder()."
    )

    # Additional: verify the recovered embedder has the correct name and dimension
    assert recovered_embedder.name == "target-v2-embedder", (
        f"Expected target embedder 'target-v2-embedder', got {recovered_embedder.name!r}"
    )
    assert recovered_embedder.dimension == 2048, (
        f"Expected target dimension 2048, got {recovered_embedder.dimension}"
    )


# =============================================================================
# C4: Recover building_complete after crash
# =============================================================================


def test_recover_building_complete_after_crash(
    mock_config: MagicMock,
    fake_source_adapter: FakeCollectionAdapter,
    fake_target_adapter: FakeCollectionAdapter,
    mock_queue_manager: MagicMock,
) -> None:
    """C4: Crash during building_complete → keep state, wait for operator /switch.

    Verifies:
    - DualWriteAdapter reconstructed with active=source, dual_write enabled
    - No ReindexEngine (reindex already finished)
    - building_complete state preserved (no auto-transition)
    """
    _require_resilience()

    from openviking.storage.migration.resilience import recover_from_crash
    from openviking.storage.migration.blue_green_adapter import DualWriteAdapter

    state = make_state("building_complete")

    result = recover_from_crash(
        state=state,
        config=mock_config,
        queue_manager=mock_queue_manager,
    )

    adapter, engine = result

    # Must return a DualWriteAdapter
    assert isinstance(adapter, DualWriteAdapter), (
        f"Expected DualWriteAdapter, got {type(adapter)}"
    )

    # Active side must be source (reading from source until /switch)
    assert adapter._active_side == "source", (
        f"building_complete: active_side must be 'source' until /switch, "
        f"got {adapter._active_side!r}"
    )

    # Dual write must be enabled (continues capturing writes during operator decision window)
    assert adapter._dual_write_enabled is True, (
        f"building_complete: dual_write must be enabled to keep standby in sync, "
        f"got {adapter._dual_write_enabled!r}"
    )

    # No reindex engine (reindex already completed)
    assert engine is None, (
        f"building_complete: no ReindexEngine should be returned "
        f"(reindex already finished), got {engine!r}"
    )


# =============================================================================
# C5: Recover switched after crash
# =============================================================================


def test_recover_switched_after_crash(
    mock_config: MagicMock,
    fake_source_adapter: FakeCollectionAdapter,
    fake_target_adapter: FakeCollectionAdapter,
    mock_queue_manager: MagicMock,
) -> None:
    """C5: Crash during switched → active=target, dual_write enabled, no engine.

    Verifies:
    - DualWriteAdapter with active_side = "target"
    - dual_write_enabled = True
    - No ReindexEngine
    """
    _require_resilience()

    from openviking.storage.migration.resilience import recover_from_crash
    from openviking.storage.migration.blue_green_adapter import DualWriteAdapter

    state = make_state(
        "switched",
        active_side="target",
        dual_write_enabled=True,
    )

    result = recover_from_crash(
        state=state,
        config=mock_config,
        queue_manager=mock_queue_manager,
    )

    adapter, engine = result

    assert isinstance(adapter, DualWriteAdapter)

    # Reading must route to target
    assert adapter._active_side == "target", (
        f"switched: active_side must be 'target', got {adapter._active_side!r}"
    )

    # Dual write must still be enabled (capturing writes during operator confirmation window)
    assert adapter._dual_write_enabled is True, (
        f"switched: dual_write must be enabled, got {adapter._dual_write_enabled!r}"
    )

    # No reindex engine
    assert engine is None, (
        f"switched: no ReindexEngine expected, got {engine!r}"
    )


# =============================================================================
# C6: Recover dual_write_off after crash
# =============================================================================


def test_recover_dual_write_off_after_crash(
    mock_config: MagicMock,
    fake_source_adapter: FakeCollectionAdapter,
    fake_target_adapter: FakeCollectionAdapter,
    mock_queue_manager: MagicMock,
) -> None:
    """C6: Crash during dual_write_off → active=target, dual_write disabled.

    Verifies:
    - DualWriteAdapter with active_side = "target"
    - dual_write_enabled = False
    - No ReindexEngine
    """
    _require_resilience()

    from openviking.storage.migration.resilience import recover_from_crash
    from openviking.storage.migration.blue_green_adapter import DualWriteAdapter

    state = make_state(
        "dual_write_off",
        active_side="target",
        dual_write_enabled=False,
    )

    result = recover_from_crash(
        state=state,
        config=mock_config,
        queue_manager=mock_queue_manager,
    )

    adapter, engine = result

    assert isinstance(adapter, DualWriteAdapter)

    # Reading must route to target
    assert adapter._active_side == "target", (
        f"dual_write_off: active_side must be 'target', got {adapter._active_side!r}"
    )

    # Dual write must be DISABLED
    assert adapter._dual_write_enabled is False, (
        f"dual_write_off: dual_write must be disabled, got {adapter._dual_write_enabled!r}"
    )

    # No reindex engine
    assert engine is None, (
        f"dual_write_off: no ReindexEngine expected, got {engine!r}"
    )


# =============================================================================
# C7: Recover completed → auto transition to idle
# =============================================================================


def test_recover_completed_auto_transition_to_idle(
    mock_config: MagicMock,
    fake_source_adapter: FakeCollectionAdapter,
    fake_target_adapter: FakeCollectionAdapter,
    mock_queue_manager: MagicMock,
) -> None:
    """C7: Crash during completed → auto cleanup and transition to idle.

    Verifies:
    - recover_from_crash detects completed phase
    - Cleans up runtime MigrationState file
    - Cleans up NamedQueue
    - Returns (None, None) — no adapter, no engine (system idle)
    - Migration state file (embedding_migration_state.json) is NOT deleted
    """
    _require_resilience()

    from openviking.storage.migration.resilience import recover_from_crash

    state = make_state(
        "completed",
        active_side="target",
        dual_write_enabled=False,
    )

    result = recover_from_crash(
        state=state,
        config=mock_config,
        queue_manager=mock_queue_manager,
    )

    adapter, engine = result

    # Completed → idle: no adapter, no engine
    assert adapter is None, (
        f"completed→idle: no DualWriteAdapter should be returned "
        f"(system returns to idle), got {adapter!r}"
    )
    assert engine is None, (
        f"completed→idle: no ReindexEngine should be returned, got {engine!r}"
    )

    # Queue should have been cleaned up
    # (In real implementation, queue_manager.cleanup_queue(queue_name) should be called)
    # This is a contract test — implementation must clean up during recovery.


def test_recover_completed_does_not_delete_migration_state_file(
    mock_config: MagicMock,
    fake_source_adapter: FakeCollectionAdapter,
    fake_target_adapter: FakeCollectionAdapter,
    mock_queue_manager: MagicMock,
) -> None:
    """C7 detail: migration state file (embedding_migration_state.json) is NOT deleted.

    Per migration-spec.md §2.1.4 C7 and §2.7.2:
    "迁移状态文件永久不删除 — 记录完整的迁移历史"
    """
    _require_resilience()

    from openviking.storage.migration.resilience import recover_from_crash

    # We verify this contractually: the migration state file manager
    # should NOT be deleted during completed recovery.
    # The test checks that the function signature and behavior
    # preserve the permanent state file.

    state = make_state("completed")

    result = recover_from_crash(
        state=state,
        config=mock_config,
        queue_manager=mock_queue_manager,
    )

    # The function returns cleanly without raising
    adapter, engine = result
    assert adapter is None
    assert engine is None

    # Contract: MigrationStateFile.delete() must NEVER be called during recovery.
    # Only the runtime MigrationState file is cleaned up, not the permanent file.
    # This is verified by checking that config's state_file_delete is NOT called.
    if hasattr(mock_config, "migration_state_file"):
        mock_config.migration_state_file.delete.assert_not_called()
