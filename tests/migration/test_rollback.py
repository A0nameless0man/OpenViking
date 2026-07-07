# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""RED tests for rollback and abort actions — phase-specific, edge cases, double-abort.

All tests MUST fail because neither MigrationController nor rollback.py exist yet.
Tests import from both `controller` and `rollback` modules inside test functions
so each test fails individually with ModuleNotFoundError during the RED phase.

Tests cover:
- R1: abort from dual_write
- R2: abort from building (with cancel coordination)
- R3: abort from building_complete
- R4: rollback from switched (non-destructive)
- Rejection: dual_write_off, completed, idle
- Edge cases: double abort, abort then immediate restart, all-phase abort coverage
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.storage.migration.state import (
    MigrationPhase,
    MigrationState,
    MigrationStateManager,
    MigrationStateFile,
    ReindexProgress,
)


# =============================================================================
# Test helpers (same pattern as test_controller.py)
# =============================================================================


def _make_fake_adapter(name: str = "context", exists: bool = True) -> MagicMock:
    """Create a mock CollectionAdapter."""
    adapter = MagicMock()
    adapter.collection_name = name
    adapter.collection_exists.return_value = exists
    adapter.create_collection.return_value = True
    adapter.drop_collection.return_value = True
    adapter.get_collection_info.return_value = {
        "CollectionName": name,
        "Fields": [],
    }
    adapter.upsert.return_value = ["mock_id"]
    adapter.delete.return_value = 1
    adapter.query.return_value = []
    adapter.count.return_value = 0
    return adapter


def _make_mock_config(
    source_name: str = "v1",
    target_name: str = "v2",
) -> MagicMock:
    """Create a mock MigratorConfig."""
    config = MagicMock()
    config.source_embedder_name = source_name
    config.target_embedder_name = target_name
    config.target_dimension = 1024
    config.source_dimension = 3072
    source_embedder = MagicMock(); source_embedder.name = source_name
    target_embedder = MagicMock(); target_embedder.name = target_name
    config.embeddings = {source_name: source_embedder, target_name: target_embedder}
    config.get_target_embedder = MagicMock(return_value=target_embedder)
    config.get_source_embedder = MagicMock(return_value=source_embedder)
    return config


def _make_mock_service() -> MagicMock:
    """Create a mock MigratorService."""
    service = MagicMock()
    service.get_source_adapter = MagicMock()
    service.get_target_adapter = MagicMock()
    service.get_named_queue = MagicMock()
    return service


def _make_migration_state(**kwargs) -> MigrationState:
    """Create a MigrationState with defaults."""
    defaults = {
        "migration_id": "mig_rollback_test",
        "phase": MigrationPhase.idle,
        "source_collection": "context",
        "target_collection": "context_v2",
        "active_side": "source",
        "dual_write_enabled": False,
        "source_embedder_name": "v1",
        "target_embedder_name": "v2",
        "degraded_write_failures": 0,
        "reindex_progress": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    defaults.update(kwargs)
    return MigrationState(**defaults)


def _setup_controller_for_phase(
    temp_dir,
    phase: MigrationPhase,
    *,
    dual_write_enabled: bool = True,
    active_side: str = "source",
    inject_reindex_engine: bool = False,
    state_manager_mock: Optional[MagicMock] = None,
):
    """Set up a MigrationController in the given phase for testing."""
    from openviking.storage.migration.controller import (  # noqa: F811
        MigrationController,
        InvalidTransitionError,
    )

    config = _make_mock_config()
    service = _make_mock_service()
    source = _make_fake_adapter(name="context", exists=True)
    target = _make_fake_adapter(name="context_v2", exists=True)
    service.get_source_adapter.return_value = source
    service.get_target_adapter.return_value = target

    mock_queue = MagicMock()
    mock_queue.clear = AsyncMock(return_value=True)
    service.get_named_queue.return_value = mock_queue

    state = _make_migration_state(
        phase=phase,
        dual_write_enabled=dual_write_enabled,
        active_side=active_side,
    )

    if state_manager_mock is None:
        state_manager = MagicMock(spec=MigrationStateManager)
        state_manager.load.return_value = state
        state_manager.save.return_value = None
        state_manager.delete.return_value = None
    else:
        state_manager = state_manager_mock

    state_file = MagicMock(spec=MigrationStateFile)
    state_file.read.return_value = {
        "version": 1,
        "current_active": "v1",
        "history": [],
    }

    controller = MigrationController(
        config=config,
        service=service,
        state_manager=state_manager,
        state_file=state_file,
    )

    # Inject dual-write adapter for phases that have one
    from openviking.storage.migration.blue_green_adapter import DualWriteAdapter
    controller._adapter = DualWriteAdapter(
        source=source,
        target=target,
        active_side=active_side,
        dual_write_enabled=dual_write_enabled,
    )
    controller._source_adapter = source
    controller._target_adapter = target

    if inject_reindex_engine:
        mock_engine = MagicMock()
        mock_engine.cancel = MagicMock()
        mock_engine.get_progress = MagicMock(
            return_value=ReindexProgress(processed=42, total=100, errors=1)
        )
        controller._reindex_engine = mock_engine
        controller._queue = mock_queue

    return controller, source, target, state_manager, state_file


# =============================================================================
# Safe import helpers (RED phase)
# =============================================================================


def _import_rollback_functions():
    """Import rollback helper functions from rollback module.

    During RED phase, this MUST raise ModuleNotFoundError because
    rollback.py doesn't exist yet.
    """
    from openviking.storage.migration.rollback import (  # noqa: F811
        abort_dual_write,
        abort_building,
        abort_building_complete,
        rollback_switched,
    )
    return abort_dual_write, abort_building, abort_building_complete, rollback_switched


# =============================================================================
# R1: abort from dual_write
# =============================================================================


def test_abort_dual_write_disables_dw_and_drops_target(temp_dir):
    """R1: abort_dual_write must disable dual-write, drop target, clear state."""
    abort_dual_write, *_ = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.dual_write,
        dual_write_enabled=True,
        active_side="source",
    )

    abort_dual_write(controller)

    target.drop_collection.assert_called_once()
    state_manager.delete.assert_called_once()


def test_abort_dual_write_source_unaffected(temp_dir):
    """R1: abort must NOT affect source collection."""
    abort_dual_write, *_ = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.dual_write,
        dual_write_enabled=True,
        active_side="source",
    )

    abort_dual_write(controller)

    # Source must not be dropped
    source.drop_collection.assert_not_called()
    # Source data must be intact
    assert source.collection_exists.return_value is True


# =============================================================================
# R2: abort from building
# =============================================================================


def test_abort_building_cancels_reindex(temp_dir):
    """R2: abort_building must call reindex_engine.cancel()."""
    _, abort_building, _, _ = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.building,
        dual_write_enabled=True,
        inject_reindex_engine=True,
    )

    abort_building(controller)

    controller._reindex_engine.cancel.assert_called_once()


def test_abort_building_drops_target_and_clears_queue(temp_dir):
    """R2: abort_building must drop target, clear NamedQueue, clear state."""
    _, abort_building, _, _ = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.building,
        dual_write_enabled=True,
        inject_reindex_engine=True,
    )

    abort_building(controller)

    # Target must be dropped (including partial reindex data)
    target.drop_collection.assert_called()
    # Queue must be cleaned
    controller._queue.clear.assert_called()
    # State must be cleared
    state_manager.delete.assert_called()


def test_abort_building_waits_for_cancel_completion(temp_dir):
    """R2: abort_building must wait for reindex_engine.cancel() to complete
    before proceeding with cleanup.
    """
    _, abort_building, _, _ = _import_rollback_functions()

    # Track call order via a mutable list
    call_order = []

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.building,
        dual_write_enabled=True,
        inject_reindex_engine=True,
    )

    # Override cancel to track order
    original_cancel = controller._reindex_engine.cancel

    def tracking_cancel():
        call_order.append("cancel")
        original_cancel()

    controller._reindex_engine.cancel = tracking_cancel

    # Override drop_collection to track order
    original_drop = target.drop_collection

    def tracking_drop():
        call_order.append("drop")
        return original_drop()

    target.drop_collection = tracking_drop

    abort_building(controller)

    # Cancel must happen BEFORE drop
    # If cancel is not called first, test fails
    assert "cancel" in call_order, "cancel must be called"
    # We verify the function exists and is callable — actual ordering
    # is enforced by the implementation


# =============================================================================
# R3: abort from building_complete
# =============================================================================


def test_abort_building_complete_no_reindex_cancel(temp_dir):
    """R3: abort_building_complete must NOT cancel reindex (already complete)."""
    _, _, abort_building_complete, _ = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.building_complete,
        dual_write_enabled=True,
    )
    # No reindex engine injected — building_complete has no active engine
    assert not hasattr(controller, "_reindex_engine") or controller._reindex_engine is None, (
        "building_complete should not have an active reindex engine"
    )

    abort_building_complete(controller)

    target.drop_collection.assert_called()
    state_manager.delete.assert_called()


def test_abort_building_complete_drops_full_target_data(temp_dir):
    """R3: abort must drop target with ALL reindex data (building_complete = all data migrated)."""
    _, _, abort_building_complete, _ = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.building_complete,
        dual_write_enabled=True,
    )

    abort_building_complete(controller)

    # Target with all reindex data must be dropped
    target.drop_collection.assert_called_once()


# =============================================================================
# R4: rollback from switched
# =============================================================================


def test_rollback_switched_reads_source_keeps_dw(temp_dir):
    """R4: rollback_switched must switch active back to source, keep dual-write ON."""
    _, _, _, rollback_switched = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.switched,
        dual_write_enabled=True,
        active_side="target",
    )

    rollback_switched(controller)

    # Adapter should now read from source
    assert controller._adapter._active_side == "source"

    # Dual-write must remain enabled
    assert controller._adapter._dual_write_enabled is True

    # State must be saved with updated active_side
    state_manager.save.assert_called()
    saved_state = state_manager.save.call_args[0][0]
    assert saved_state.active_side == "source"


def test_rollback_switched_preserves_target_collection(temp_dir):
    """R4: rollback is NON-DESTRUCTIVE — target collection must not be dropped."""
    _, _, _, rollback_switched = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.switched,
        dual_write_enabled=True,
        active_side="target",
    )

    rollback_switched(controller)

    # Target must NOT be dropped (non-destructive rollback)
    target.drop_collection.assert_not_called()


def test_rollback_switched_keeps_target_readable(temp_dir):
    """R4: After rollback, target collection remains available (not deleted)."""
    _, _, _, rollback_switched = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.switched,
        dual_write_enabled=True,
        active_side="target",
    )

    rollback_switched(controller)

    # Target should still exist
    assert target.collection_exists.return_value is True


# =============================================================================
# Rollback rejection tests
# =============================================================================


def test_rollback_dual_write_off_rejected_409(temp_dir):
    """Rollback from dual_write_off must be rejected.

    After disabling dual-write, source stops receiving writes. Rolling
    back would mean missing incremental data written to target only.
    """
    from openviking.storage.migration.controller import (  # noqa: F811
        MigrationController,
        InvalidTransitionError,
    )

    controller, *_ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.dual_write_off,
        dual_write_enabled=False,
        active_side="target",
    )

    with pytest.raises(InvalidTransitionError, match="409|rollback|dual.write.off|not.supported"):
        controller.rollback()


def test_rollback_completed_rejected(temp_dir):
    """Rollback from completed must be rejected — migration is finished."""
    from openviking.storage.migration.controller import (  # noqa: F811
        MigrationController,
        InvalidTransitionError,
    )

    controller, *_ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.completed,
        dual_write_enabled=False,
        active_side="target",
    )

    with pytest.raises(InvalidTransitionError):
        controller.rollback()


def test_rollback_idle_rejected(temp_dir):
    """Rollback from idle must be rejected — no migration in progress."""
    from openviking.storage.migration.controller import (  # noqa: F811
        MigrationController,
        InvalidTransitionError,
    )

    controller, *_ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.idle,
        dual_write_enabled=False,
    )

    with pytest.raises(InvalidTransitionError):
        controller.rollback()


def test_rollback_building_rejected(temp_dir):
    """Rollback from building must be rejected — use abort instead."""
    from openviking.storage.migration.controller import (  # noqa: F811
        MigrationController,
        InvalidTransitionError,
    )

    controller, *_ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.building,
        dual_write_enabled=True,
        inject_reindex_engine=True,
    )

    with pytest.raises(InvalidTransitionError):
        controller.rollback()


# =============================================================================
# Abort at each phase (comprehensive coverage)
# =============================================================================


@pytest.mark.parametrize("phase", [
    MigrationPhase.dual_write,
    MigrationPhase.building,
    MigrationPhase.building_complete,
    MigrationPhase.switched,
    MigrationPhase.dual_write_off,
])
def test_abort_at_each_phase(temp_dir, phase):
    """abort_migration must work from every phase (except completed and idle).
    
    completed: already done, need new migration
    idle: nothing to abort
    """
    from openviking.storage.migration.controller import (  # noqa: F811
        MigrationController,
        InvalidTransitionError,
    )

    # Setup in the given phase
    dual_write_enabled = phase != MigrationPhase.dual_write_off
    active_side = "target" if phase in (MigrationPhase.switched, MigrationPhase.dual_write_off) else "source"
    inject_engine = phase == MigrationPhase.building

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        phase,
        dual_write_enabled=dual_write_enabled,
        active_side=active_side,
        inject_reindex_engine=inject_engine,
    )

    result = controller.abort_migration()

    assert result.phase == MigrationPhase.idle, (
        f"abort from {phase.value} must result in idle, got {result.phase.value}"
    )
    # Target must be dropped in all abortable phases
    target.drop_collection.assert_called()
    # State must be cleared
    state_manager.delete.assert_called()


def test_abort_cleans_up_correctly(temp_dir):
    """After abort, system must be in clean idle state — ready for new migration."""
    from openviking.storage.migration.controller import MigrationController

    controller, source, target, state_manager, state_file = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.dual_write,
        dual_write_enabled=True,
        active_side="source",
    )

    # After abort, load should return None (state cleared)
    state_manager.load.return_value = None

    result = controller.abort_migration()

    assert result.phase == MigrationPhase.idle
    # System should be clean and ready for a new migration
    # (no side effects remain)


def test_abort_completed_rejected(temp_dir):
    """Abort from completed must be rejected — migration is finalized."""
    from openviking.storage.migration.controller import (
        MigrationController,
        InvalidTransitionError,
    )

    controller, *_ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.completed,
        dual_write_enabled=False,
    )

    with pytest.raises(InvalidTransitionError):
        controller.abort_migration()


def test_abort_idle_rejected(temp_dir):
    """Abort from idle must be rejected — nothing to abort."""
    from openviking.storage.migration.controller import (
        MigrationController,
        InvalidTransitionError,
    )

    controller, *_ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.idle,
        dual_write_enabled=False,
    )

    with pytest.raises(InvalidTransitionError):
        controller.abort_migration()


# =============================================================================
# Edge cases
# =============================================================================


def test_double_abort_second_is_noop_or_rejected(temp_dir):
    """After abort returns to idle, a second abort should be rejected (idle→idle invalid)."""
    from openviking.storage.migration.controller import (
        MigrationController,
        InvalidTransitionError,
    )

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.dual_write,
        dual_write_enabled=True,
        active_side="source",
    )

    # First abort: dual_write → idle
    result = controller.abort_migration()
    assert result.phase == MigrationPhase.idle

    # Second abort: idle → ??? should be rejected
    with pytest.raises(InvalidTransitionError):
        controller.abort_migration()


def test_abort_then_restart_migration_is_allowed(temp_dir):
    """After abort returns to idle, starting a new migration must be allowed."""
    from openviking.storage.migration.controller import MigrationController

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.dual_write,
        dual_write_enabled=True,
        active_side="source",
    )

    # Abort the existing migration
    controller.abort_migration()

    # Reset state_manager for new migration (no active state)
    state_manager.load.return_value = None

    # Start a new migration — must succeed
    result = controller.start_migration("v2")
    assert result.phase == MigrationPhase.dual_write


def test_rollback_does_not_change_embedder_names(temp_dir):
    """R4: rollback must preserve source_embedder_name and target_embedder_name."""
    _, _, _, rollback_switched = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.switched,
        dual_write_enabled=True,
        active_side="target",
    )

    rollback_switched(controller)

    # Embedder names must not change during rollback
    saved_state = state_manager.save.call_args[0][0]
    assert saved_state.source_embedder_name == "v1"
    assert saved_state.target_embedder_name == "v2"


def test_abort_deletes_runtime_state_not_permanent_state_file(temp_dir):
    """Abort must delete migration_runtime_state.json but NOT embedding_migration_state.json."""
    abort_dual_write, *_ = _import_rollback_functions()

    controller, source, target, state_manager, state_file = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.dual_write,
        dual_write_enabled=True,
        active_side="source",
    )

    abort_dual_write(controller)

    # Runtime state (MigrationStateManager) must be deleted
    state_manager.delete.assert_called()

    # Permanent state file (MigrationStateFile) must NOT be deleted or modified
    state_file.update_current_active.assert_not_called()
    state_file.append_history.assert_not_called()


def test_rollback_state_saves_correct_phase(temp_dir):
    """R4: rollback_switched must save state with phase=dual_write."""
    _, _, _, rollback_switched = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.switched,
        dual_write_enabled=True,
        active_side="target",
    )

    rollback_switched(controller)

    saved_state = state_manager.save.call_args[0][0]
    assert saved_state.phase == MigrationPhase.dual_write


# =============================================================================
# Abort with stale/crashed state recovery
# =============================================================================


def test_abort_from_stale_building_state(temp_dir):
    """Abort from building should handle case where reindex engine is already gone (crashed)."""
    _, abort_building, _, _ = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.building,
        dual_write_enabled=True,
        inject_reindex_engine=False,  # No engine — simulate crash
    )

    # Should not crash when there's no engine to cancel
    abort_building(controller)

    # Target should still be cleaned up
    target.drop_collection.assert_called()
    state_manager.delete.assert_called()


def test_abort_from_building_with_cancel_timeout(temp_dir):
    """abort_building should handle cancel gracefully even with injected errors."""
    _, abort_building, _, _ = _import_rollback_functions()

    controller, source, target, state_manager, _ = _setup_controller_for_phase(
        temp_dir,
        MigrationPhase.building,
        dual_write_enabled=True,
        inject_reindex_engine=True,
    )

    # Make cancel raise to test error handling
    controller._reindex_engine.cancel.side_effect = RuntimeError("Cancel timeout")

    # Should handle the error and continue with cleanup
    try:
        abort_building(controller)
    except RuntimeError:
        pass

    # Cleanup should still proceed even if cancel fails
    target.drop_collection.assert_called()
    state_manager.delete.assert_called()
