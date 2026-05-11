"""Optimization job stores for local and TimescaleDB-backed API runs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, Optional, Protocol

from flexihome.api.serialization import json_safe


STATUS_COUNTS = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


class OptimizationStore(Protocol):
    def create(self, optimization_id: str, request: Dict) -> StoredJob:
        ...

    def mark_running(self, optimization_id: str) -> None:
        ...

    def complete(self, optimization_id: str, result: Dict) -> None:
        ...

    def fail(self, optimization_id: str, error: str) -> None:
        ...

    def get(self, optimization_id: str) -> Optional[StoredJob]:
        ...

    def list(self) -> list[StoredJob]:
        ...

    def metrics(self) -> Dict[str, int]:
        ...

    def health(self) -> Dict[str, str]:
        ...


class InMemoryOptimizationStore:
    def __init__(self, fallback_reason: str | None = None) -> None:
        self._jobs: Dict[str, StoredJob] = {}
        self._lock = RLock()
        self.fallback_reason = fallback_reason

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
        counts = dict(STATUS_COUNTS)
        with self._lock:
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
        counts["total"] = sum(counts.values())
        return counts

    def health(self) -> Dict[str, str]:
        payload = {
            "job_store": "in_memory",
            "timescaledb": "unavailable" if self.fallback_reason else "not_configured",
        }
        if self.fallback_reason:
            payload["timescaledb_error"] = self.fallback_reason
        return payload

    def _update(self, optimization_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[optimization_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = utc_now()


class TimescaleOptimizationStore:
    """Persist optimizer jobs in PostgreSQL with a TimescaleDB event hypertable."""

    def __init__(
        self,
        dsn: str,
        schema: str = "coverly",
        jobs_table: str = "optimization_jobs",
        events_table: str = "optimization_job_events",
    ) -> None:
        self.dsn = dsn
        self.schema = _validate_identifier(schema, "schema")
        self.jobs_table = _validate_identifier(jobs_table, "jobs table")
        self.events_table = _validate_identifier(events_table, "events table")
        self._load_driver()
        self._initialize_schema()

    def create(self, optimization_id: str, request: Dict) -> StoredJob:
        now = utc_now()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    self._sql.SQL(
                        """
                        INSERT INTO {} (
                            optimization_id, status, request, result, error, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, NULL, NULL, %s, %s)
                        RETURNING optimization_id, status, request, result, error, created_at, updated_at
                        """
                    ).format(self._jobs_ref()),
                    (optimization_id, "queued", self._jsonb(request), now, now),
                )
                row = cur.fetchone()
                self._record_event(cur, optimization_id, "queued", "created", {"request": request}, now)
        return self._row_to_job(row)

    def mark_running(self, optimization_id: str) -> None:
        self._update_job(optimization_id, "running", event_type="running", error=None)

    def complete(self, optimization_id: str, result: Dict) -> None:
        payload = {
            "summary": result.get("summary", {}),
            "row_counts": result.get("row_counts", {}),
        }
        self._update_job(
            optimization_id,
            "completed",
            result=result,
            error=None,
            event_type="completed",
            event_payload=payload,
        )

    def fail(self, optimization_id: str, error: str) -> None:
        self._update_job(
            optimization_id,
            "failed",
            error=error,
            event_type="failed",
            event_payload={"error": error},
        )

    def get(self, optimization_id: str) -> Optional[StoredJob]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    self._sql.SQL(
                        """
                        SELECT optimization_id, status, request, result, error, created_at, updated_at
                        FROM {}
                        WHERE optimization_id = %s
                        """
                    ).format(self._jobs_ref()),
                    (optimization_id,),
                )
                row = cur.fetchone()
        return self._row_to_job(row) if row else None

    def list(self) -> list[StoredJob]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    self._sql.SQL(
                        """
                        SELECT optimization_id, status, request, result, error, created_at, updated_at
                        FROM {}
                        ORDER BY created_at DESC
                        LIMIT 1000
                        """
                    ).format(self._jobs_ref())
                )
                rows = cur.fetchall()
        return [self._row_to_job(row) for row in rows]

    def metrics(self) -> Dict[str, int]:
        counts = dict(STATUS_COUNTS)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    self._sql.SQL("SELECT status, COUNT(*) AS count FROM {} GROUP BY status").format(
                        self._jobs_ref()
                    )
                )
                for row in cur.fetchall():
                    counts[str(row["status"])] = int(row["count"])
        counts["total"] = sum(counts.values())
        return counts

    def health(self) -> Dict[str, str]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                    row = cur.fetchone()
            return {
                "job_store": "timescaledb",
                "timescaledb": "connected",
                "timescaledb_extension": str(row["extversion"]) if row else "missing",
                "timescaledb_schema": self.schema,
            }
        except Exception as exc:
            return {
                "job_store": "timescaledb",
                "timescaledb": "unavailable",
                "timescaledb_error": _short_error(exc),
                "timescaledb_schema": self.schema,
            }

    def _update_job(
        self,
        optimization_id: str,
        status: str,
        result: Dict | None = None,
        error: str | None = None,
        event_type: str = "updated",
        event_payload: Dict | None = None,
    ) -> None:
        now = utc_now()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    self._sql.SQL(
                        """
                        UPDATE {}
                        SET status = %s,
                            result = COALESCE(%s, result),
                            error = %s,
                            updated_at = %s
                        WHERE optimization_id = %s
                        """
                    ).format(self._jobs_ref()),
                    (
                        status,
                        self._jsonb(result) if result is not None else None,
                        error,
                        now,
                        optimization_id,
                    ),
                )
                if cur.rowcount != 1:
                    raise KeyError(f"Optimization job not found: {optimization_id}")
                self._record_event(
                    cur,
                    optimization_id,
                    status,
                    event_type,
                    event_payload or {},
                    now,
                )

    def _record_event(
        self,
        cur,
        optimization_id: str,
        status: str,
        event_type: str,
        payload: Dict,
        created_at: datetime,
    ) -> None:
        cur.execute(
            self._sql.SQL(
                """
                INSERT INTO {} (optimization_id, status, event_type, payload, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """
            ).format(self._events_ref()),
            (optimization_id, status, event_type, self._jsonb(payload), created_at),
        )

    def _initialize_schema(self) -> None:
        with self._connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    self._sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        self._sql.Identifier(self.schema)
                    )
                )
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                cur.execute(
                    self._sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            optimization_id TEXT PRIMARY KEY,
                            status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
                            request JSONB NOT NULL,
                            result JSONB,
                            error TEXT,
                            created_at TIMESTAMPTZ NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL
                        )
                        """
                    ).format(self._jobs_ref())
                )
                cur.execute(
                    self._sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            optimization_id TEXT NOT NULL,
                            status TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            payload JSONB NOT NULL DEFAULT jsonb_build_object(),
                            created_at TIMESTAMPTZ NOT NULL
                        )
                        """
                    ).format(self._events_ref())
                )
                cur.execute(
                    "SELECT create_hypertable(%s, 'created_at', if_not_exists => TRUE)",
                    (f"{self.schema}.{self.events_table}",),
                )
                cur.execute(
                    self._sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (status)").format(
                        self._index_ref("idx_flexihome_jobs_status"),
                        self._jobs_ref(),
                    )
                )
                cur.execute(
                    self._sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (optimization_id, created_at DESC)").format(
                        self._index_ref("idx_flexihome_job_events_id_time"),
                        self._events_ref(),
                    )
                )

    def _connect(self, autocommit: bool = False):
        return self._psycopg.connect(
            self.dsn,
            autocommit=autocommit,
            connect_timeout=_connect_timeout_seconds(),
            row_factory=self._dict_row,
        )

    def _jobs_ref(self):
        return self._qualified_ref(self.jobs_table)

    def _events_ref(self):
        return self._qualified_ref(self.events_table)

    def _index_ref(self, index_name: str):
        return self._sql.Identifier(index_name)

    def _qualified_ref(self, name: str):
        return self._sql.SQL("{}.{}").format(
            self._sql.Identifier(self.schema),
            self._sql.Identifier(name),
        )

    def _row_to_job(self, row) -> StoredJob:
        return StoredJob(
            optimization_id=str(row["optimization_id"]),
            status=str(row["status"]),
            request=dict(row["request"] or {}),
            result=dict(row["result"] or {}) if row["result"] is not None else None,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _jsonb(self, payload: Dict):
        return self._Jsonb(json_safe(payload))

    def _load_driver(self) -> None:
        try:
            import psycopg
            from psycopg import sql
            from psycopg.rows import dict_row
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise RuntimeError(
                "TimescaleDB support requires psycopg. Install requirements.txt first."
            ) from exc
        self._psycopg = psycopg
        self._sql = sql
        self._dict_row = dict_row
        self._Jsonb = Jsonb


def build_optimization_store() -> OptimizationStore:
    dsn = os.getenv("FLEXIHOME_TIMESCALE_DSN", "").strip()
    if not dsn:
        return InMemoryOptimizationStore()
    try:
        return TimescaleOptimizationStore(
            dsn=dsn,
            schema=os.getenv("FLEXIHOME_TIMESCALE_SCHEMA", "coverly").strip() or "coverly",
            jobs_table=os.getenv("FLEXIHOME_TIMESCALE_JOBS_TABLE", "optimization_jobs").strip()
            or "optimization_jobs",
            events_table=os.getenv("FLEXIHOME_TIMESCALE_EVENTS_TABLE", "optimization_job_events").strip()
            or "optimization_job_events",
        )
    except Exception as exc:
        if os.getenv("FLEXIHOME_TIMESCALE_STRICT", "").lower() in {"1", "true", "yes"}:
            raise
        return InMemoryOptimizationStore(fallback_reason=_short_error(exc))


def _validate_identifier(value: str, label: str) -> str:
    if not IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid {label} identifier: {value!r}")
    return value


def _short_error(exc: Exception) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return message[:240]


def _connect_timeout_seconds() -> int:
    try:
        return max(1, int(os.getenv("FLEXIHOME_TIMESCALE_CONNECT_TIMEOUT", "5")))
    except ValueError:
        return 5

