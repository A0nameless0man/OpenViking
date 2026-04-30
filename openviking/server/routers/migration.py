# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Migration endpoints for OpenViking HTTP Server.

Provides the REST API for embedding migration lifecycle management:
start → dual_write → building → building_complete → switched →
dual_write_off → completed → idle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext, Role
from openviking.server.models import Response
from openviking.storage.migration.controller import (
    InvalidTransitionError,
    MigrationController,
)
from openviking.storage.migration.state import (
    MigrationStateFile,
    MigrationStateManager,
)
from openviking_cli.exceptions import ConflictError, InvalidArgumentError, PermissionDeniedError
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import get_openviking_config

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/migration", tags=["migration"])

# ---------------------------------------------------------------------------
# Auth dependency — FastAPI-ready callable that enforces ADMIN role
# ---------------------------------------------------------------------------


async def _require_admin_role(
    ctx: RequestContext = Depends(get_request_context),
) -> RequestContext:
    """FastAPI dependency: require ADMIN or ROOT role."""
    if ctx.role not in (Role.ADMIN, Role.ROOT):
        raise PermissionDeniedError("Requires role: admin or root")
    return ctx

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class StartMigrationRequest(BaseModel):
    """Request to begin an embedding migration."""

    target_name: str


class FinishMigrationRequest(BaseModel):
    """Request to finalise a completed migration."""

    confirm_cleanup: bool = False


class MigrationStatusResponse(BaseModel):
    """Snapshot of the current migration status."""

    model_config = ConfigDict(extra="allow")

    migration_id: str = ""
    phase: str = ""
    active_side: str = ""
    source_collection: str = ""
    target_collection: str = ""
    dual_write_enabled: bool = False
    source_embedder_name: str = ""
    target_embedder_name: str = ""
    degraded_write_failures: int = 0
    started_at: str = ""
    updated_at: str = ""
    reindex_progress: Optional[dict] = None


class TargetInfo(BaseModel):
    """Information about an available target embedding configuration."""

    name: str
    provider: str
    model: str
    dimension: int


class TargetsResponse(BaseModel):
    """List of available migration target embedding configurations."""

    targets: list[TargetInfo]


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

_controller: Optional[MigrationController] = None


class _MigrationServiceAdapter:
    """Adapts OpenVikingService to the interface expected by MigrationController.

    Provides ``get_source_adapter()``, ``get_target_adapter()``,
    and ``get_named_queue()`` using the underlying VectorDB and QueueFS
    infrastructure.
    """

    def __init__(self, service: Any, ov_config: Any) -> None:
        self._service = service
        self._ov_config = ov_config

    def get_source_adapter(self) -> Any:
        """Return the existing source-collection adapter."""
        vikingdb_manager = getattr(self._service, "vikingdb_manager", None)
        if vikingdb_manager is not None:
            return vikingdb_manager
        # Fallback: create a fresh adapter from config
        from openviking.storage.vectordb_adapters.factory import (
            create_collection_adapter,
        )
        return create_collection_adapter(self._ov_config.storage.vectordb)

    def get_target_adapter(self) -> Any:
        """Create a new adapter pointing at a target collection."""
        from copy import deepcopy

        from openviking.storage.vectordb_adapters.factory import (
            create_collection_adapter,
        )

        target_config = deepcopy(self._ov_config.storage.vectordb)
        source_name = target_config.name or "context"
        target_config.name = f"{source_name}_migration_target"
        return create_collection_adapter(target_config)

    def get_named_queue(self, name: str) -> Any:
        """Return a NamedQueue from the QueueManager."""
        queue_manager = getattr(self._service, "_queue_manager", None)
        if queue_manager is None:
            raise RuntimeError("QueueManager not initialised")
        return queue_manager.get_queue(name)


class _MigrationConfigAdapter:
    """Adapts OpenVikingConfig to the interface expected by MigrationController.

    Exposes ``get_target_embedder()``, ``source_embedder_name``, and
    ``queue_name`` so the controller can resolve embedder instances by name.
    """

    def __init__(self, ov_config: Any) -> None:
        self._ov_config = ov_config

    @property
    def source_embedder_name(self) -> str:
        """Return the name of the currently-active source embedder."""
        return "default"

    @property
    def queue_name(self) -> str:
        """Name of the reindex queue."""
        return "reindex"

    def get_target_embedder(self, target_name: str) -> Any:
        """Resolve the named target embedder from ``ov_config.embeddings``."""
        if target_name not in self._ov_config.embeddings:
            raise KeyError(
                f"Target embedding '{target_name}' not found in "
                f"embeddings config. Available: "
                f"{list(self._ov_config.embeddings.keys())}"
            )
        return self._ov_config.embeddings[target_name].get_embedder()


def _get_controller() -> MigrationController:
    """Get or lazily build the singleton MigrationController.

    The controller is cached after first creation so that its lifecycle
    (phase, adapters, reindex engine) survives across requests.
    """
    global _controller
    if _controller is not None:
        return _controller

    service = get_service()
    ov_config = get_openviking_config()

    # Persistence directories (lives under workspace)
    workspace = ov_config.storage.workspace
    from pathlib import Path

    state_dir = Path(workspace) / ".migration" / "state"
    config_dir = Path(workspace) / ".migration" / "config"

    state_manager = MigrationStateManager(str(state_dir))
    state_file = MigrationStateFile(str(config_dir))

    migration_config = _MigrationConfigAdapter(ov_config)
    migration_service = _MigrationServiceAdapter(service, ov_config)

    _controller = MigrationController(
        config=migration_config,
        service=migration_service,
        state_manager=state_manager,
        state_file=state_file,
    )
    return _controller


# ---------------------------------------------------------------------------
# Helper — build a status response from the controller's dict snapshot
# ---------------------------------------------------------------------------


def _status_response(status_dict: Dict[str, Any]) -> Response:
    """Wrap the controller status dict in a standard Response."""
    return Response(
        status="ok",
        result=MigrationStatusResponse(**status_dict).model_dump(),
    )


def _handle_controller_error(exc: Exception) -> None:
    """Map controller exceptions to OpenVikingError subclasses.

    Raises:
        ConflictError: For InvalidTransitionError (HTTP 409)
        InvalidArgumentError: For ValueError / KeyError (HTTP 400)
    """
    if isinstance(exc, InvalidTransitionError):
        raise ConflictError(str(exc))
    if isinstance(exc, (ValueError, KeyError)):
        raise InvalidArgumentError(str(exc))
    # Re-raise other exceptions unchanged (caught by general handler → 500)
    raise


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start")
async def start_migration(
    body: StartMigrationRequest,
    ctx: RequestContext = Depends(_require_admin_role),
):
    """Begin migration: transition from idle to dual_write."""
    try:
        ctrl = _get_controller()
        state = ctrl.start_migration(body.target_name)
        return _status_response(state.to_dict())
    except Exception as exc:
        _handle_controller_error(exc)


@router.get("/status")
async def get_migration_status(
    _ctx: RequestContext = Depends(get_request_context),
):
    """Return the current migration phase and full status."""
    ctrl = _get_controller()
    status_dict = ctrl.get_status()
    return _status_response(status_dict)


@router.get("/targets")
async def list_targets(
    _ctx: RequestContext = Depends(get_request_context),
):
    """List available target embedding configurations for migration."""
    ov_config = get_openviking_config()
    targets: list[TargetInfo] = []
    for name, emb_cfg in ov_config.embeddings.items():
        provider = emb_cfg.dense.provider if emb_cfg.dense and emb_cfg.dense.provider else "unknown"
        model = emb_cfg.dense.model if emb_cfg.dense and emb_cfg.dense.model else "unknown"
        targets.append(
            TargetInfo(
                name=name,
                provider=provider,
                model=model,
                dimension=emb_cfg.dimension,
            )
        )
    return Response(
        status="ok",
        result=TargetsResponse(targets=targets).model_dump(),
    )


@router.post("/build")
async def begin_building(
    ctx: RequestContext = Depends(_require_admin_role),
):
    """Transition to the building phase — start background reindex."""
    try:
        ctrl = _get_controller()
        state = ctrl.begin_building()
        return _status_response(state.to_dict())
    except Exception as exc:
        _handle_controller_error(exc)


@router.post("/switch")
async def confirm_switch(
    ctx: RequestContext = Depends(_require_admin_role),
):
    """Switch the active read side to the target embedder."""
    try:
        ctrl = _get_controller()
        state = ctrl.confirm_switch()
        return _status_response(state.to_dict())
    except Exception as exc:
        _handle_controller_error(exc)


@router.post("/disable-dual-write")
async def disable_dual_write(
    ctx: RequestContext = Depends(_require_admin_role),
):
    """Disable dual-write — writes now go exclusively to target."""
    try:
        ctrl = _get_controller()
        state = ctrl.disable_dual_write()
        return _status_response(state.to_dict())
    except Exception as exc:
        _handle_controller_error(exc)


@router.post("/finish")
async def finish_migration(
    body: FinishMigrationRequest,
    ctx: RequestContext = Depends(_require_admin_role),
):
    """Finalise migration: update permanent state, clean runtime artefacts."""
    try:
        ctrl = _get_controller()
        state = ctrl.finish_migration(confirm_cleanup=body.confirm_cleanup)
        return _status_response(state.to_dict())
    except Exception as exc:
        _handle_controller_error(exc)


@router.post("/abort")
async def abort_migration(
    ctx: RequestContext = Depends(_require_admin_role),
):
    """Abort the migration — revert to idle regardless of current phase."""
    try:
        ctrl = _get_controller()
        state = ctrl.abort_migration()
        return _status_response(state.to_dict())
    except Exception as exc:
        _handle_controller_error(exc)


@router.post("/rollback")
async def rollback_migration(
    ctx: RequestContext = Depends(_require_admin_role),
):
    """Non-destructive rollback: switch active side back to source."""
    try:
        ctrl = _get_controller()
        state = ctrl.rollback()
        return _status_response(state.to_dict())
    except Exception as exc:
        _handle_controller_error(exc)
