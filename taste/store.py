"""Storage — G1 permutation: SQLite WAL, embeddings as packed f32 BLOBs.

At corpus scale (hundreds–thousands of standards), cosine over loaded
vectors is faster than maintaining an ANN index — B1 is plain cosine by
design. sqlite-vec can be introduced later without changing callers.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import threading
from datetime import UTC
from pathlib import Path

from .models import (
    Category,
    DecisionRecord,
    LayerState,
    Polarity,
    Provenance,
    Standard,
)


class HoneBudgetExhausted(Exception):
    """A task has used its hone calls. The cap is per task, across segments."""


DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "taste1.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-64000;

CREATE TABLE IF NOT EXISTS standards (
  id          TEXT PRIMARY KEY,
  profile_id  TEXT NOT NULL,
  category    TEXT NOT NULL,
  polarity    TEXT NOT NULL,
  directive   TEXT NOT NULL,
  rationale   TEXT NOT NULL DEFAULT '',
  confidence  REAL NOT NULL DEFAULT 0.5,
  provenance  TEXT NOT NULL,
  state       TEXT NOT NULL DEFAULT 'provisional',
  supersedes  TEXT NOT NULL DEFAULT '[]',
  created     TEXT NOT NULL,
  updated     TEXT NOT NULL,
  variant_meta TEXT NOT NULL DEFAULT '{}',
  embedding   BLOB
);

CREATE INDEX IF NOT EXISTS idx_standards_scope
  ON standards(profile_id, state, category, polarity);

CREATE TABLE IF NOT EXISTS decisions (
  id          TEXT PRIMARY KEY,
  profile_id  TEXT NOT NULL,
  task_ctx    TEXT NOT NULL DEFAULT '{}',
  candidates  TEXT NOT NULL DEFAULT '[]',
  pick        TEXT NOT NULL,
  rationale   TEXT NOT NULL,
  agent_role  TEXT NOT NULL DEFAULT '',
  provenance  TEXT NOT NULL DEFAULT 'owner_pick',
  ts          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inferred_queue (
  id          TEXT PRIMARY KEY,
  profile_id  TEXT NOT NULL,
  kind        TEXT NOT NULL,           -- mirror | answer | amendment
  payload     TEXT NOT NULL,           -- json
  ts          TEXT NOT NULL,
  resolved    INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS standards_fts USING fts5(
  id UNINDEXED, directive, rationale
);

CREATE TABLE IF NOT EXISTS hone_budgets (
  profile_id    TEXT NOT NULL,
  task_key      TEXT NOT NULL,
  reservations  INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (profile_id, task_key)
);
"""


def _pack(vec: list[float] | None) -> bytes | None:
    return None if vec is None else struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class Store:
    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path or os.environ.get("TASTE1_DB", DEFAULT_DB))
        path.parent.mkdir(parents=True, exist_ok=True)
        # single-process service: one connection, writes serialized by lock,
        # cross-thread because the ASGI worker owns a different thread
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------ standards
    def put_standard(self, s: Standard, embedding: list[float] | None = None):
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO standards"
                " (id, profile_id, category, polarity, directive, rationale,"
                "  confidence, provenance, state, supersedes, created, updated,"
                "  variant_meta, embedding)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (s.id, s.profile_id, s.category.value, s.polarity.value,
                 s.directive, s.rationale, s.confidence, s.provenance.value,
                 s.state.value, json.dumps(s.supersedes),
                 s.created.isoformat(), s.updated.isoformat(),
                 json.dumps(s.variant_meta), _pack(embedding)),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO standards_fts (id, directive, rationale)"
                " VALUES (?,?,?)", (s.id, s.directive, s.rationale))
            self.db.commit()

    def get_standard(self, sid: str) -> Standard | None:
        row = self.db.execute(
            "SELECT * FROM standards WHERE id=?", (sid,)).fetchone()
        return self._row_to_standard(row) if row else None

    def list_standards(self, profile_id: str = "owner",
                       state: LayerState | None = None,
                       category: Category | None = None,
                       polarity: Polarity | None = None,
                       with_vectors: bool = False
                       ) -> list[Standard] | list[tuple[Standard, list[float] | None]]:
        q = "SELECT * FROM standards WHERE profile_id=?"
        args: list = [profile_id]
        if state:
            q += " AND state=?"
            args.append(state.value)
        if category:
            q += " AND category=?"
            args.append(category.value)
        if polarity:
            q += " AND polarity=?"
            args.append(polarity.value)
        rows = self.db.execute(q, args).fetchall()
        if with_vectors:
            return [(self._row_to_standard(r), _unpack(r["embedding"]))
                    for r in rows]
        return [self._row_to_standard(r) for r in rows]

    def set_state(self, sid: str, state: LayerState):
        with self._lock:
            self.db.execute(
                "UPDATE standards SET state=?, updated=datetime('now') WHERE id=?",
                (state.value, sid))
            self.db.commit()

    def set_variant_meta(self, sid: str, meta: dict):
        with self._lock:
            self.db.execute(
                "UPDATE standards SET variant_meta=?, updated=datetime('now')"
                " WHERE id=?",
                (json.dumps(meta), sid))
            self.db.commit()

    def set_rationale(self, sid: str, rationale: str):
        with self._lock:
            self.db.execute(
                "UPDATE standards SET rationale=?, updated=datetime('now')"
                " WHERE id=?", (rationale, sid))
            self.db.execute(
                "UPDATE standards_fts SET rationale=? WHERE id=?",
                (rationale, sid))
            self.db.commit()

    def reserve_hone(self, profile_id: str, task_key: str, limit: int = 15) -> int:
        """Count one hone call. Raise once this task is past `limit`."""
        with self._lock:
            self.db.execute(
                "INSERT OR IGNORE INTO hone_budgets"
                " (profile_id, task_key, reservations) VALUES (?,?,0)",
                (profile_id, task_key))
            row = self.db.execute(
                "SELECT reservations FROM hone_budgets"
                " WHERE profile_id=? AND task_key=?",
                (profile_id, task_key)).fetchone()
            current = int(row["reservations"])
            if current >= limit:
                self.db.rollback()
                raise HoneBudgetExhausted("The task hone budget is exhausted.")
            reservation = current + 1
            self.db.execute(
                "UPDATE hone_budgets SET reservations=?"
                " WHERE profile_id=? AND task_key=?",
                (reservation, profile_id, task_key))
            self.db.commit()
            return reservation

    def set_confidence(self, sid: str, conf: float):
        with self._lock:
            self.db.execute(
                "UPDATE standards SET confidence=?, updated=datetime('now')"
                " WHERE id=?", (conf, sid))
            self.db.commit()

    @staticmethod
    def _row_to_standard(r: sqlite3.Row) -> Standard:
        return Standard(
            id=r["id"], profile_id=r["profile_id"],
            category=Category(r["category"]), polarity=Polarity(r["polarity"]),
            directive=r["directive"], rationale=r["rationale"],
            confidence=r["confidence"], provenance=Provenance(r["provenance"]),
            state=LayerState(r["state"]),
            supersedes=json.loads(r["supersedes"]),
            created=r["created"], updated=r["updated"],
            variant_meta=json.loads(r["variant_meta"]),
        )

    # ------------------------------------------------------------ decisions
    def put_decision(self, d: DecisionRecord):
        with self._lock:
            self.db.execute(
                "INSERT INTO decisions"
                " (id, profile_id, task_ctx, candidates, pick, rationale,"
                "  agent_role, provenance, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (d.id, d.profile_id, json.dumps(d.task_ctx),
                 json.dumps([c.model_dump() for c in d.candidates]),
                 d.pick, d.rationale, d.agent_role, d.provenance,
                 d.ts.isoformat()),
            )
            self.db.commit()

    def list_decisions(self, profile_id: str = "owner") -> list[DecisionRecord]:
        rows = self.db.execute(
            "SELECT * FROM decisions WHERE profile_id=? ORDER BY ts",
            (profile_id,)).fetchall()
        return [DecisionRecord(
            id=r["id"], profile_id=r["profile_id"],
            task_ctx=json.loads(r["task_ctx"]),
            candidates=[],  # hydrated on demand — list view stays light
            pick=r["pick"], rationale=r["rationale"],
            agent_role=r["agent_role"], provenance=r["provenance"],
            ts=r["ts"]) for r in rows]

    # ------------------------------------------------------------ inferred queue
    def enqueue_inferred(self, iid: str, profile_id: str, kind: str,
                         payload: dict):
        from datetime import datetime
        with self._lock:
            self.db.execute(
                "INSERT INTO inferred_queue (id, profile_id, kind, payload, ts)"
                " VALUES (?,?,?,?,?)",
                (iid, profile_id, kind, json.dumps(payload),
                 datetime.now(UTC).isoformat()))
            self.db.commit()

    def pending_inferred(self, profile_id: str = "owner") -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM inferred_queue WHERE profile_id=? AND resolved=0"
            " ORDER BY ts", (profile_id,)).fetchall()
        return [{"id": r["id"], "kind": r["kind"],
                 "payload": json.loads(r["payload"]), "ts": r["ts"]}
                for r in rows]

    def resolve_inferred(self, iid: str):
        with self._lock:
            self.db.execute(
                "UPDATE inferred_queue SET resolved=1 WHERE id=?", (iid,))
            self.db.commit()
