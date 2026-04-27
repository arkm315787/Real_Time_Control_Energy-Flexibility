"""Small process-local optimization job store.

This gives the first API layer useful async semantics without requiring a
database during local development. The methods mirror what a TimescaleDB-backed
repository would expose, so replacing this later is intentionally boring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StoredJob:
    optimization_id: str
    status: str
    request: Dict
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict] = None
    error: Optional[str] = None


class InMemoryOptimizationStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, StoredJob] = {}
        self._lock = RLock()

    def create(self, optimization_id: str, request: Dict) -> StoredJob:
        now = utc_now()
        job = StoredJob(
            optimization_id=optimization_id,
            status="queued",
            request=request,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[optimization_id] = job
        return job

    def mark_running(self, optimization_id: str) -> None:
        self._update(optimization_id, status="running", error=None)

    def complete(self, optimization_id: str, result: Dict) -> None:
        self._update(optimization_id, status="completed", result=result, error=None)

    def fail(self, optimization_id: str, error: str) -> None:
        self._update(optimization_id, status="failed", error=error)

    def get(self, optimization_id: str) -> Optional[StoredJob]:
        with self._lock:
            return self._jobs.get(optimization_id)

    def list(self) -> list[StoredJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def metrics(self) -> Dict[str, int]:
        counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
        with self._lock:
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
        counts["total"] = sum(counts.values())
        return counts

    def _update(self, optimization_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[optimization_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = utc_now()

