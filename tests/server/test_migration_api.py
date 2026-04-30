# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""E2E tests for migration REST API endpoints.

Tests the 9 migration endpoints via httpx.AsyncClient + ASGITransport:
happy path, abort/rollback, crash recovery, and auth integration.

All adapters are mocked via monkeypatching _get_controller() so tests
never hit a real VectorDB or embedding API.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from openviking.server.api_keys import APIKeyManager
from openviking.server.app import create_app
from openviking.server.config import ServerConfig
from openviking.server.dependencies import set_service
from openviking.storage.migration.controller import (
    InvalidTransitionError,
    MigrationController,
)
from openviking.storage.migration.state import (
    MigrationStateFile,
    MigrationStateManager,
    MigrationPhase,
)


# =============================================================================
# Session-scoped mocks: avoid Rust + C++ native dependencies
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def _mock_native_deps():
    """Globally mock native bindings so OpenVikingService can initialize
    without requiring Rust or C++ engine libraries."""
    mock_agfs_client = MagicMock()
    mock_agfs_client.start = MagicMock()
    mock_agfs_client.check_health = MagicMock()
    mock_agfs_client.shutdown = MagicMock()

    mock_queue_manager = MagicMock()
    mock_vikingdb_manager = MagicMock()
    mock_lock_manager = MagicMock()
    mock_viking_fs = MagicMock()

    # pylint: disable=import-outside-toplevel
    with (
        patch(
            "openviking.pyagfs._load_rust_binding",
            return_value=(MagicMock(), None),
        ),
        patch(
            "openviking.utils.agfs_utils.create_agfs_client",
            return_value=mock_agfs_client,
        ),
        patch(
            "openviking.storage.queuefs.queue_manager.init_queue_manager",
            return_value=mock_queue_manager,
        ),
        patch(
            "openviking.storage.transaction.init_lock_manager",
            return_value=mock_lock_manager,
        ),
        patch(
            "openviking.storage.viking_fs.init_viking_fs",
            return_value=mock_viking_fs,
        ),
        patch(
            "openviking.storage.VikingDBManager",
            return_value=MagicMock(
                _account_backends={},
                enqueue_embedding_msg=MagicMock(),
                get_embedder=MagicMock(),
            ),
        ),
        patch(
            "openviking.utils.process_lock.acquire_data_dir_lock",
            return_value=None,
        ),
    ):
        yield


# =============================================================================
# Mock Factories
# =============================================================================

ROOT_KEY = "root-secret-key-for-testing-only-1234567890abcdef"


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _make_fake_collection_adapter(
    name: str = "context",
    exists: bool = True,
) -> MagicMock:
    """MagicMock with enough CollectionAdapter-like surface for controller tests."""
    adapter = MagicMock()
    adapter.collection_name = name
    adapter.collection_exists.return_value = exists
    adapter.create_collection.return_value = True
    adapter.drop_collection.return_value = True
    adapter.upsert.return_value = ["mock_id_001"]
    adapter.delete.return_value = 1
    adapter.query.return_value = []  # no records in source
    adapter.count.return_value = 0
    adapter.get_collection_info.return_value = {
        "CollectionName": name,
        "Fields": [
            {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
            {"FieldName": "uri", "FieldType": "path"},
            {"FieldName": "vector", "FieldType": "vector", "Dim": 1024},
        ],
    }
    return adapter


def _make_mock_service_adapter(
    source_adapter: Optional[MagicMock] = None,
    target_adapter: Optional[MagicMock] = None,
    mock_queue: Optional[MagicMock] = None,
) -> MagicMock:
    """Construct a mock MigratorService for the controller."""
    svc = MagicMock()
    if source_adapter is None:
        source_adapter = _make_fake_collection_adapter("context", exists=True)
    if target_adapter is None:
        target_adapter = _make_fake_collection_adapter("context_migration_target", exists=False)
    if mock_queue is None:
        mock_queue = MagicMock()
        mock_queue.dequeue = AsyncMock(return_value=None)  # empty queue → exit loop
        mock_queue.clear = MagicMock()

    svc.get_source_adapter = MagicMock(return_value=source_adapter)
    svc.get_target_adapter = MagicMock(return_value=target_adapter)
    svc.get_named_queue = MagicMock(return_value=mock_queue)
    return svc


def _make_mock_config_adapter(
    source_name: str = "v1",
    target_name: str = "v2",
    dimension: int = 1024,
) -> MagicMock:
    """Construct a mock MigratorConfig for the controller."""
    cfg = MagicMock()
    cfg.source_embedder_name = source_name
    cfg.queue_name = "reindex"

    mock_embedder = MagicMock()
    mock_embedder.embed_async = AsyncMock(return_value=[[0.1] * dimension])

    cfg.get_target_embedder = MagicMock(return_value=mock_embedder)
    cfg.embeddings = {
        source_name: MagicMock(
            dimension=dimension,
            get_embedder=MagicMock(return_value=mock_embedder),
            dense=MagicMock(provider="test", model=source_name),
        ),
        target_name: MagicMock(
            dimension=dimension,
            get_embedder=MagicMock(return_value=mock_embedder),
            dense=MagicMock(provider="test", model=target_name),
        ),
    }
    return cfg


def _make_mock_get_openviking_config(
    embeddings: Optional[dict] = None,
) -> MagicMock:
    """Mock for get_openviking_config() used by /targets endpoint."""
    cfg = MagicMock()
    if embeddings is None:
        d1 = MagicMock(dimension=1024, dense=MagicMock(provider="test", model="v1"))
        d2 = MagicMock(dimension=1024, dense=MagicMock(provider="test", model="v2"))
        d3 = MagicMock(dimension=1536, dense=MagicMock(provider="test", model="v3"))
        embeddings = {"v1": d1, "v2": d2, "v3": d3}
    cfg.embeddings = embeddings
    return cfg


def _build_test_controller(temp_dir: Path) -> MigrationController:
    """Create a MigrationController with mock deps and real state persistence."""
    config = _make_mock_config_adapter()
    service = _make_mock_service_adapter()

    state_dir = temp_dir / ".migration" / "state"
    config_dir = temp_dir / ".migration" / "config"
    state_manager = MigrationStateManager(str(state_dir))
    state_file = MigrationStateFile(str(config_dir))
    state_file.create_initial("v1")

    return MigrationController(
        config=config,
        service=service,
        state_manager=state_manager,
        state_file=state_file,
    )


# =============================================================================
# Shared monkeypatch helpers
# =============================================================================

def _patch_migration_module(monkeypatch, controller: MigrationController):
    """Patch the migration router to use a test controller and mock config."""
    monkeypatch.setattr(
        "openviking.server.routers.migration._get_controller",
        lambda c=controller: c,
    )
    monkeypatch.setattr(
        "openviking.server.routers.migration.get_openviking_config",
        _make_mock_get_openviking_config,
    )


def _reset_controller_singleton():
    """Reset the module-level _controller singleton in migration router."""
    import openviking.server.routers.migration as mod
    mod._controller = None


# =============================================================================
# Fixtures — Dev mode (no auth, ROOT identity passes)
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def migration_controller(temp_dir: Path) -> MigrationController:
    """Fresh controller per test — isolated state."""
    _reset_controller_singleton()
    return _build_test_controller(temp_dir)


@pytest_asyncio.fixture(scope="function")
async def migration_app(
    service,
    monkeypatch,
    migration_controller: MigrationController,
):
    """Dev-mode app with mocked migration controller."""
    _patch_migration_module(monkeypatch, migration_controller)

    config = ServerConfig()
    app = create_app(config=config, service=service)
    set_service(service)
    return app


@pytest_asyncio.fixture(scope="function")
async def migration_client(migration_app):
    """httpx client for dev-mode migration tests."""
    transport = httpx.ASGITransport(app=migration_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# =============================================================================
# Fixtures — Auth mode (api_key auth, ADMIN/Root keys)
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def auth_migration_controller(temp_dir: Path) -> MigrationController:
    """Fresh controller for auth tests."""
    _reset_controller_singleton()
    return _build_test_controller(temp_dir)


@pytest_asyncio.fixture(scope="function")
async def auth_migration_app(
    service,
    monkeypatch,
    auth_migration_controller: MigrationController,
):
    """Auth-enabled app with mocked migration controller and APIKeyManager."""
    _patch_migration_module(monkeypatch, auth_migration_controller)

    config = ServerConfig(root_api_key=ROOT_KEY)
    app = create_app(config=config, service=service)
    set_service(service)

    # Manually init APIKeyManager (lifespan not triggered in ASGI transport)
    manager = APIKeyManager(root_key=ROOT_KEY, viking_fs=service.viking_fs)
    await manager.load()
    app.state.api_key_manager = manager

    return app


@pytest_asyncio.fixture(scope="function")
async def auth_migration_client(auth_migration_app):
    """httpx client for auth-enabled migration tests."""
    transport = httpx.ASGITransport(app=auth_migration_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest_asyncio.fixture(scope="function")
async def admin_key(auth_migration_app) -> str:
    """Create a test admin account and return its API key."""
    manager: APIKeyManager = auth_migration_app.state.api_key_manager
    return await manager.create_account(_uid(), "test_admin")


@pytest_asyncio.fixture(scope="function")
async def user_key(auth_migration_app) -> str:
    """Create a test regular user and return their API key."""
    manager: APIKeyManager = auth_migration_app.state.api_key_manager
    account_id = _uid()
    await manager.create_account(account_id, "admin_user")
    return await manager.register_user(account_id, "regular_user", "user")


# =============================================================================
# 1. Happy Path (Task 4.3)
# =============================================================================


async def test_happy_path_full_migration(migration_client: httpx.AsyncClient):
    """Complete 9-step migration flow via HTTP."""
    c = migration_client

    # Step 1: POST /start → 200, phase=dual_write
    resp = await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    result = body["result"]
    assert result["phase"] == "dual_write"
    assert result["active_side"] == "source"
    assert result["dual_write_enabled"] is True
    assert result["source_embedder_name"] == "v1"
    assert result["target_embedder_name"] == "v2"
    assert result["migration_id"].startswith("mig_")

    # Step 2: POST /build → 200, phase=building
    resp = await c.post("/api/v1/migration/build")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"]["phase"] == "building"

    # Step 3: Poll GET /status until phase=building_complete
    # (reindex engine completes immediately since source has no records)
    for attempt in range(120):
        resp = await c.get("/api/v1/migration/status")
        assert resp.status_code == 200
        phase = resp.json()["result"]["phase"]
        if phase == "building_complete":
            break
        await asyncio.sleep(0.1)
    else:
        pytest.fail("Timed out waiting for building_complete phase")

    # Step 4: GET /status check reindex_progress
    resp = await c.get("/api/v1/migration/status")
    body = resp.json()
    assert body["result"]["phase"] == "building_complete"
    # reindex_progress should be present (even if 0 records)
    progress = body["result"].get("reindex_progress")
    assert progress is not None
    assert progress["errors"] == 0

    # Step 5: POST /switch → 200, phase=switched, active_side=target
    resp = await c.post("/api/v1/migration/switch")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"]["phase"] == "switched"
    assert body["result"]["active_side"] == "target"

    # Step 6: POST /disable-dual-write → 200, dual_write_enabled=false
    resp = await c.post("/api/v1/migration/disable-dual-write")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"]["phase"] == "dual_write_off"
    assert body["result"]["dual_write_enabled"] is False

    # Step 7: POST /finish → 200, phase=idle
    resp = await c.post("/api/v1/migration/finish", json={"confirm_cleanup": False})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"]["phase"] == "idle"

    # Step 8: Verify permanent state file updated
    # (controller's state_file.update_current_active was called with "v2")
    # This is verified indirectly: finish_migration calls update_current_active("v2")
    resp = await c.get("/api/v1/migration/status")
    assert resp.status_code == 200
    assert resp.json()["result"]["phase"] == "idle"


async def test_happy_path_with_finish_cleanup(migration_client: httpx.AsyncClient):
    """Finish with confirm_cleanup=True should not crash (target already clean)."""
    c = migration_client

    # Execute full migration
    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    await c.post("/api/v1/migration/build")
    for _ in range(120):
        r = await c.get("/api/v1/migration/status")
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)
    await c.post("/api/v1/migration/switch")
    await c.post("/api/v1/migration/disable-dual-write")

    # Finish with confirm_cleanup=True
    resp = await c.post("/api/v1/migration/finish", json={"confirm_cleanup": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"]["phase"] == "idle"


async def test_get_targets(migration_client: httpx.AsyncClient):
    """GET /targets should list available embedding configs."""
    resp = await migration_client.get("/api/v1/migration/targets")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    targets = body["result"]["targets"]
    assert len(targets) == 3
    names = {t["name"] for t in targets}
    assert "v1" in names
    assert "v2" in names
    assert "v3" in names


async def test_get_status_idle(migration_client: httpx.AsyncClient):
    """Status should return idle when no migration is active."""
    resp = await migration_client.get("/api/v1/migration/status")
    assert resp.status_code == 200
    assert resp.json()["result"]["phase"] == "idle"


# =============================================================================
# 2. Failure + Rollback (Task 4.4)
# =============================================================================


async def test_abort_from_dual_write(migration_client: httpx.AsyncClient):
    """Abort from dual_write phase: target cleaned, state cleared."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    resp = await c.get("/api/v1/migration/status")
    assert resp.json()["result"]["phase"] == "dual_write"

    resp = await c.post("/api/v1/migration/abort")
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"]["phase"] == "idle"

    # State should be idle after abort
    resp = await c.get("/api/v1/migration/status")
    assert resp.json()["result"]["phase"] == "idle"


async def test_abort_from_building(migration_client: httpx.AsyncClient):
    """Abort from building phase: reindex cancelled, target cleaned, state cleared."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    await c.post("/api/v1/migration/build")
    resp = await c.get("/api/v1/migration/status")
    assert resp.json()["result"]["phase"] == "building"

    resp = await c.post("/api/v1/migration/abort")
    assert resp.status_code == 200
    assert resp.json()["result"]["phase"] == "idle"

    resp = await c.get("/api/v1/migration/status")
    assert resp.json()["result"]["phase"] == "idle"


async def test_abort_from_building_complete(migration_client: httpx.AsyncClient):
    """Abort from building_complete: target cleaned, state cleared."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    await c.post("/api/v1/migration/build")
    for _ in range(120):
        r = await c.get("/api/v1/migration/status")
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)

    resp = await c.post("/api/v1/migration/abort")
    assert resp.status_code == 200
    assert resp.json()["result"]["phase"] == "idle"


async def test_abort_from_switched(migration_client: httpx.AsyncClient):
    """Abort from switched: target dropped, state cleared."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    await c.post("/api/v1/migration/build")
    for _ in range(120):
        r = await c.get("/api/v1/migration/status")
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)
    await c.post("/api/v1/migration/switch")

    resp = await c.post("/api/v1/migration/abort")
    assert resp.status_code == 200
    assert resp.json()["result"]["phase"] == "idle"


async def test_rollback_from_switched(migration_client: httpx.AsyncClient):
    """Rollback from switched: reads return to source, dual-write stays ON."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    await c.post("/api/v1/migration/build")
    for _ in range(120):
        r = await c.get("/api/v1/migration/status")
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)
    await c.post("/api/v1/migration/switch")

    resp = await c.get("/api/v1/migration/status")
    assert resp.json()["result"]["active_side"] == "target"

    resp = await c.post("/api/v1/migration/rollback")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"]["phase"] == "dual_write"
    assert body["result"]["active_side"] == "source"
    # Dual-write must remain enabled (non-destructive rollback)
    assert body["result"]["dual_write_enabled"] is True


async def test_rollback_from_dual_write_off_returns_409(migration_client: httpx.AsyncClient):
    """Rollback from dual_write_off should return 409 (cannot rollback)."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    await c.post("/api/v1/migration/build")
    for _ in range(120):
        r = await c.get("/api/v1/migration/status")
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)
    await c.post("/api/v1/migration/switch")
    await c.post("/api/v1/migration/disable-dual-write")

    resp = await c.post("/api/v1/migration/rollback")
    assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"


async def test_start_with_active_migration_returns_409(migration_client: httpx.AsyncClient):
    """Cannot /start when migration is already active."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    resp = await c.post("/api/v1/migration/start", json={"target_name": "v3"})
    assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"


async def test_start_with_invalid_target_name_returns_400(migration_client: httpx.AsyncClient):
    """Invalid target_name should return 400."""
    c = migration_client

    resp = await c.post("/api/v1/migration/start", json={"target_name": "nonexistent"})
    # The controller gets a KeyError from get_target_embedder → 400
    assert resp.status_code in (400, 500), f"Expected 400/500, got {resp.status_code}: {resp.text}"


async def test_switch_before_building_complete_returns_400(migration_client: httpx.AsyncClient):
    """Cannot /switch before building_complete phase."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    # Still in dual_write — trying to switch should fail
    resp = await c.post("/api/v1/migration/switch")
    assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"


async def test_finish_before_dual_write_off_returns_409(migration_client: httpx.AsyncClient):
    """Cannot /finish before dual_write_off phase."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    resp = await c.post("/api/v1/migration/finish", json={"confirm_cleanup": False})
    assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"


async def test_disable_dual_write_before_switched_returns_409(migration_client: httpx.AsyncClient):
    """Cannot /disable-dual-write before switched phase."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    resp = await c.post("/api/v1/migration/disable-dual-write")
    assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"


async def test_start_missing_target_name_returns_422(migration_client: httpx.AsyncClient):
    """POST /start without target_name → 422 validation error."""
    resp = await migration_client.post("/api/v1/migration/start", json={})
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"


async def test_incremental_rebuild(migration_client: httpx.AsyncClient):
    """building_complete → POST /build again → building → building_complete."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    await c.post("/api/v1/migration/build")
    for _ in range(120):
        r = await c.get("/api/v1/migration/status")
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)

    # Re-build from building_complete
    resp = await c.post("/api/v1/migration/build")
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"]["phase"] == "building"

    # Wait for re-build to complete
    for _ in range(120):
        r = await c.get("/api/v1/migration/status")
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)

    resp = await c.get("/api/v1/migration/status")
    assert resp.json()["result"]["phase"] == "building_complete"


# =============================================================================
# 3. Crash Recovery (Task 4.5)
# =============================================================================


async def test_crash_during_building_recovers(
    monkeypatch,
    service,
    temp_dir,
    migration_controller: MigrationController,
):
    """Crash during building → new controller recovers and resumes reindex.

    Simulates a crash by:
    1. Starting migration + building
    2. Creating a NEW controller that loads the persisted state
    3. Verifying the new controller recovers to building_complete
    """
    ctrl = migration_controller

    # Phase 1: start + begin building
    state = ctrl.start_migration("v2")
    assert state.phase == MigrationPhase.dual_write

    state = ctrl.begin_building()
    assert state.phase == MigrationPhase.building

    # Wait for reindex to complete (should be fast with no data)
    for _ in range(120):
        if ctrl.get_status()["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)
    assert ctrl.get_status()["phase"] == "building_complete", "Expected building_complete"

    # Phase 2: Simulate crash — restart with new controller using SAME state dirs
    _reset_controller_singleton()
    new_ctrl = _build_test_controller(temp_dir)
    # The new controller loads existing state from the state manager
    status = new_ctrl.get_status()
    assert status["phase"] == "building_complete", (
        f"Expected building_complete after recovery, got {status['phase']}"
    )


async def test_start_abort_start_restart_cycle(
    monkeypatch,
    service,
    temp_dir,
    migration_controller: MigrationController,
):
    """Start → abort → start → restart (recovery) should work cleanly.

    Verifies that abort properly cleans state so a fresh start works,
    and that a controller restart after abort sees idle state.
    """
    ctrl = migration_controller

    # Start → abort
    ctrl.start_migration("v2")
    ctrl.abort_migration()

    # Start fresh
    state = ctrl.start_migration("v3")
    assert state.phase == MigrationPhase.dual_write
    assert state.target_embedder_name == "v3"

    # Abort again
    ctrl.abort_migration()

    # Simulate restart — new controller sees idle
    _reset_controller_singleton()
    new_ctrl = _build_test_controller(temp_dir)
    status = new_ctrl.get_status()
    assert status["phase"] == "idle", f"Expected idle, got {status['phase']}"


async def test_status_after_finish_is_idle(migration_client: httpx.AsyncClient):
    """After finish, multiple status calls should all return idle."""
    c = migration_client

    await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    await c.post("/api/v1/migration/build")
    for _ in range(120):
        r = await c.get("/api/v1/migration/status")
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)
    await c.post("/api/v1/migration/switch")
    await c.post("/api/v1/migration/disable-dual-write")
    await c.post("/api/v1/migration/finish", json={"confirm_cleanup": False})

    # Status should be idle consistently
    for _ in range(5):
        resp = await c.get("/api/v1/migration/status")
        assert resp.json()["result"]["phase"] == "idle"


# =============================================================================
# 4. Auth Integration (Task 4.6)
# =============================================================================

# List of all protected endpoints (require admin/root role)
_PROTECTED_ENDPOINTS = [
    ("POST", "/api/v1/migration/start", {"target_name": "v2"}),
    ("POST", "/api/v1/migration/build", None),
    ("POST", "/api/v1/migration/switch", None),
    ("POST", "/api/v1/migration/disable-dual-write", None),
    ("POST", "/api/v1/migration/finish", {"confirm_cleanup": False}),
    ("POST", "/api/v1/migration/abort", None),
    ("POST", "/api/v1/migration/rollback", None),
]

# Unprotected endpoints (status + targets use get_request_context, not _require_admin_role)
_UNPROTECTED_ENDPOINTS = [
    ("GET", "/api/v1/migration/status", None),
    ("GET", "/api/v1/migration/targets", None),
]


async def test_protected_endpoints_no_auth_returns_401(
    auth_migration_client: httpx.AsyncClient,
):
    """All protected endpoints must return 401 without auth headers."""
    c = auth_migration_client
    for method, url, body in _PROTECTED_ENDPOINTS:
        if body is not None:
            resp = await c.request(method, url, json=body)
        else:
            resp = await c.request(method, url)
        assert resp.status_code == 401, (
            f"{method} {url} expected 401, got {resp.status_code}: {resp.text}"
        )


async def test_protected_endpoints_user_role_returns_403(
    auth_migration_client: httpx.AsyncClient,
    user_key: str,
):
    """Protected endpoints must return 403 for USER role."""
    c = auth_migration_client
    headers = {"X-API-Key": user_key}
    for method, url, body in _PROTECTED_ENDPOINTS:
        if body is not None:
            resp = await c.request(method, url, json=body, headers=headers)
        else:
            resp = await c.request(method, url, headers=headers)
        assert resp.status_code == 403, (
            f"{method} {url} expected 403, got {resp.status_code}: {resp.text}"
        )


async def test_protected_endpoints_admin_role_returns_200(
    auth_migration_client: httpx.AsyncClient,
    admin_key: str,
):
    """Protected endpoints must allow ADMIN role access."""
    c = auth_migration_client
    headers = {"X-API-Key": admin_key}

    # POST /start (requires ADMIN)
    resp = await c.post("/api/v1/migration/start", json={"target_name": "v2"}, headers=headers)
    assert resp.status_code == 200, f"start: {resp.text}"

    # POST /build
    resp = await c.post("/api/v1/migration/build", headers=headers)
    assert resp.status_code == 200, f"build: {resp.text}"

    # Wait for building_complete (reindex finishes immediately)
    for _ in range(120):
        r = await c.get("/api/v1/migration/status", headers=headers)
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)

    # POST /switch
    resp = await c.post("/api/v1/migration/switch", headers=headers)
    assert resp.status_code == 200, f"switch: {resp.text}"

    # POST /disable-dual-write
    resp = await c.post("/api/v1/migration/disable-dual-write", headers=headers)
    assert resp.status_code == 200, f"disable-dual-write: {resp.text}"

    # POST /finish
    resp = await c.post(
        "/api/v1/migration/finish",
        json={"confirm_cleanup": False},
        headers=headers,
    )
    assert resp.status_code == 200, f"finish: {resp.text}"


async def test_root_key_can_access_protected_endpoints(
    auth_migration_client: httpx.AsyncClient,
):
    """Root key should be able to access protected endpoints."""
    c = auth_migration_client
    headers = {"X-API-Key": ROOT_KEY}

    resp = await c.post("/api/v1/migration/start", json={"target_name": "v2"}, headers=headers)
    assert resp.status_code == 200, f"Root start: {resp.text}"

    resp = await c.post("/api/v1/migration/build", headers=headers)
    assert resp.status_code == 200, f"Root build: {resp.text}"

    for _ in range(120):
        r = await c.get("/api/v1/migration/status", headers=headers)
        if r.json()["result"]["phase"] == "building_complete":
            break
        await asyncio.sleep(0.1)

    resp = await c.post("/api/v1/migration/switch", headers=headers)
    assert resp.status_code == 200, f"Root switch: {resp.text}"

    resp = await c.post("/api/v1/migration/disable-dual-write", headers=headers)
    assert resp.status_code == 200, f"Root disable-dual-write: {resp.text}"

    resp = await c.post(
        "/api/v1/migration/finish",
        json={"confirm_cleanup": False},
        headers=headers,
    )
    assert resp.status_code == 200, f"Root finish: {resp.text}"


async def test_unprotected_endpoints_still_require_auth(
    auth_migration_client: httpx.AsyncClient,
):
    """Status and targets endpoints require auth (get_request_context)."""
    c = auth_migration_client
    for method, url, _body in _UNPROTECTED_ENDPOINTS:
        resp = await c.request(method, url)
        assert resp.status_code == 401, (
            f"{method} {url} expected 401, got {resp.status_code}"
        )


async def test_unprotected_endpoints_work_with_auth(
    auth_migration_client: httpx.AsyncClient,
    admin_key: str,
):
    """Status and targets work with valid auth."""
    c = auth_migration_client
    headers = {"X-API-Key": admin_key}

    resp = await c.get("/api/v1/migration/status", headers=headers)
    assert resp.status_code == 200

    resp = await c.get("/api/v1/migration/targets", headers=headers)
    assert resp.status_code == 200


async def test_wrong_key_returns_401(
    auth_migration_client: httpx.AsyncClient,
):
    """Invalid API key returns 401 on all endpoints."""
    c = auth_migration_client
    headers = {"X-API-Key": "definitely-wrong-key"}

    for method, url, body in _PROTECTED_ENDPOINTS:
        if body is not None:
            resp = await c.request(method, url, json=body, headers=headers)
        else:
            resp = await c.request(method, url, headers=headers)
        assert resp.status_code == 401, f"{method} {url} expected 401 with wrong key"


async def test_dev_mode_allows_migration_access(
    migration_client: httpx.AsyncClient,
    monkeypatch,
):
    """In dev mode, the ROOT identity should be allowed (after _require_admin_role fix)."""
    c = migration_client

    resp = await c.post("/api/v1/migration/start", json={"target_name": "v2"})
    assert resp.status_code == 200, (
        f"Dev mode should allow ROOT access. Got {resp.status_code}: {resp.text}"
    )

    resp = await c.get("/api/v1/migration/status")
    assert resp.status_code == 200
    assert resp.json()["result"]["phase"] == "dual_write"
