# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Rollback helper functions for embedding migration.

These wrappers exist solely for testability — they delegate to the
corresponding ``_handle_*`` methods on ``MigrationController`` so that
the rollback/abort logic can be unit-tested in isolation without going
through the controller's public transition API.

Each function takes a ``MigrationController`` instance as its first
argument and delegates to the controller's internal handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .controller import MigrationController


def abort_dual_write(controller: "MigrationController") -> None:
    """R1: abort from dual_write — disable dw, drop target, clear state."""
    controller._handle_abort_dual_write()


def abort_building(controller: "MigrationController") -> None:
    """R2: abort from building — cancel reindex, drop target, clear queue, clear state."""
    controller._handle_abort_building()


def abort_building_complete(controller: "MigrationController") -> None:
    """R3: abort from building_complete — drop target, clear queue, clear state."""
    controller._handle_abort_building_complete()


def rollback_switched(controller: "MigrationController") -> None:
    """R4: rollback from switched — switch active back to source, keep dw, save state."""
    controller._handle_rollback_switched()
