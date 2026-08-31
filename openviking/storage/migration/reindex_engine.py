# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ReindexEngine — orchestrates reindexing from source to target embedding backend.

Scans source for URIs, filters out URIs already present in target, enqueues
missing URIs into a NamedQueue, and processes the queue by embedding records
with the target embedder and upserting them into the target adapter.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, List, Optional

from .state import ReindexProgress


class ReindexEngine:
    """Orchestrate reindexing from one embedding backend to another.

    Key semantics:
    - ``scan_source_uris``: paginated async generator yielding URI batches.
    - ``filter_missing``: synchronous batch check against target (only URIs
      NOT already in target are returned).
    - ``enqueue_uris``: deduplicate and push URIs into the NamedQueue.
    - ``process_queue``: consume the queue, embed, upsert, ack.  Uses
      ``asyncio.Semaphore`` for concurrency control and exponential backoff
      for retries.
    - ``cancel``: set an internal stop event for graceful shutdown.
    - ``get_progress``: return current ``ReindexProgress`` snapshot.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        source_adapter: Any,
        target_embedder: Any,
        target_adapter: Any,
        queue_name: str,
        max_concurrent: int = 5,
        batch_size: int = 500,
        max_retries: int = 3,
    ) -> None:
        self.source_adapter = source_adapter
        self.target_embedder = target_embedder
        self.target_adapter = target_adapter
        self.queue_name = queue_name
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.max_retries = max_retries

        self._queue: Any = None  # injected externally or set by caller
        self._cancel_event: asyncio.Event = asyncio.Event()

        # Shared mutable progress — mutated by enqueue_uris & process_queue
        self.progress: ReindexProgress = ReindexProgress()

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    async def scan_source_uris(
        self, page_size: int = 500
    ) -> AsyncIterator[List[str]]:
        """Paginated async generator yielding batches of URIs from the source.

        Uses ``output_fields=["uri"]`` to avoid loading full record payloads.
        """
        offset: int = 0
        while True:
            results: List[dict] = await asyncio.to_thread(
                self.source_adapter.query,
                filter={},
                limit=page_size,
                offset=offset,
                output_fields=["uri"],
            )
            if not results:
                break
            uris: List[str] = [r["uri"] for r in results if r.get("uri")]
            if not uris:
                break
            yield uris
            offset += page_size

    # ------------------------------------------------------------------
    # Filter (synchronous — tests call without await)
    # ------------------------------------------------------------------

    def filter_missing(self, uris_batch: List[str]) -> List[str]:
        """Return URIs from *uris_batch* that are NOT already in the target.

        Uses a single batch query with ``$in`` so we never load every target
        URI at once.
        """
        if not uris_batch:
            return []

        existing: List[dict] = self.target_adapter.query(
            filter={"uri": {"$in": uris_batch}},
            output_fields=["uri"],
            limit=len(uris_batch),
        )
        existing_set: set = {r["uri"] for r in existing if r.get("uri")}
        return [u for u in uris_batch if u not in existing_set]

    # ------------------------------------------------------------------
    # Enqueue (synchronous — tests call without await)
    # ------------------------------------------------------------------

    def enqueue_uris(self, uris: List[str]) -> int:
        """Deduplicate *uris* (via set) and enqueue each unique URI.

        Returns the count of unique URIs actually enqueued.
        """
        unique: List[str] = list(set(uris))
        self.progress.total += len(unique)
        self.progress.skipped += (len(uris) - len(unique))

        for uri in unique:
            self._queue.enqueue(uri)

        return len(unique)

    # ------------------------------------------------------------------
    # Process queue (async main loop with concurrency control)
    # ------------------------------------------------------------------

    async def process_queue(self, stop_event: asyncio.Event) -> None:
        """Consume the NamedQueue until it is empty, ``stop_event`` is set,
        or ``cancel()`` has been called.

        For each dequeued URI:
        1. Fetch the full record from *source_adapter* (by URI filter).
        2. Embed it with *target_embedder*.
        3. Upsert the record into *target_adapter*.
        4. Ack the message.

        Retries on transient failures (embed / upsert).  After exhausting
        ``max_retries`` the message is left un-acked (for RecoverStale) and
        the URI is tracked to avoid infinite re-processing loops.
        """
        semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_concurrent)
        exhausted_ids: set = set()  # msg IDs whose retries are fully used up

        while (
            not stop_event.is_set()
            and not self._cancel_event.is_set()
        ):
            msg = await self._queue.dequeue()
            if msg is None:
                break

            msg_id = msg["id"]
            if msg_id in exhausted_ids:
                # Already tried every retry — skip to prevent infinite loop
                continue

            uri: str = msg["data"]

            async with semaphore:
                success: bool = False
                for attempt in range(self.max_retries + 1):
                    try:
                        # 1. Fetch full record from source
                        data: List[dict] = await asyncio.to_thread(
                            self.source_adapter.query,
                            filter={"uri": uri},
                            limit=1,
                        )
                        if not data:
                            # Source record no longer exists — ack and move on
                            await self._queue.ack(msg_id)
                            self.progress.processed += 1
                            success = True
                            break

                        # 2. Embed with target embedder
                        embed_result: Any = await self.target_embedder.embed_async(
                            uri
                        )

                        # 3. Upsert into target
                        await asyncio.to_thread(
                            self.target_adapter.upsert, data
                        )

                        # 4. Ack the message
                        await self._queue.ack(msg_id)
                        self.progress.processed += 1
                        success = True
                        break

                    except Exception:
                        if attempt == self.max_retries:
                            # All retries exhausted — leave in queue for recovery
                            self.progress.errors += 1
                            exhausted_ids.add(msg_id)
                            success = True  # prevent re-entering inner loop
                            break
                        # Exponential backoff before retry
                        await asyncio.sleep(0.5 * (attempt + 1))

                # If we exhausted all retries, success is True but we didn't ack.
                # The `exhausted_ids` set prevents the next dequeue from re-trying.
                # Fall-through to next dequeue iteration.

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request graceful stop.  In-flight URI processing completes;
        no new URIs are dequeued after the current ones finish.
        """
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def get_progress(self) -> ReindexProgress:
        """Return a snapshot of the current reindex progress."""
        return self.progress
