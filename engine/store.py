"""Persistence, tenant isolation, schema migration and retention.

Several production requirements collapse into one place here, so they are solved
once and enforced structurally rather than by discipline.

*State must survive the process.* Beliefs and item parameters are the product.

*Tenants must not see each other.* **Every** method takes ``tenant`` first and
every statement carries ``WHERE tenant = ?``. There is no method that can read
across tenants, so there is no call site that can forget.

*Schema must be able to change.* ``CREATE TABLE IF NOT EXISTS`` is not a
migration strategy -- it works exactly once, and then the first schema change
against a live database containing real beliefs has no path forward. Migrations
are numbered, recorded, and applied in order; the ``item_tags`` index and the
``api_keys`` table arrive as migrations 2 and 3 specifically so the upgrade path
is exercised rather than assumed.

*Data must not grow forever.* ``predictions`` rows for items that were served but
never reported, and audit rows past their retention window, accumulate silently
until the disk decides the matter. ``purge`` bounds them.

Concurrency
-----------
One connection **per thread**, not one shared connection. A single shared
connection behind a mutex serialises every request -- including reads -- so the
service's effective concurrency is one regardless of how many workers are
configured. With WAL, SQLite supports concurrent readers alongside one writer,
and ``BEGIN IMMEDIATE`` takes the write lock up front so a read-modify-write on
a belief cannot interleave with another writer even across processes.

``:memory:`` is the exception: separate connections would each get their own
empty database, so in-memory mode keeps a single connection behind a lock. It is
for tests, and it is the one configuration that does not represent production.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from contracts.core import Belief, Item, TagSpace
from engine.mtor import ItemStore, MTORConfig

__all__ = ["SqliteStore", "TenantItemStore", "StoreError", "MIGRATIONS"]


class StoreError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# migrations -- append only, never edit a shipped one
# ---------------------------------------------------------------------------

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE IF NOT EXISTS tags (
        tenant   TEXT NOT NULL,
        tag      TEXT NOT NULL,
        idx      INTEGER NOT NULL,
        PRIMARY KEY (tenant, tag)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS tags_idx ON tags(tenant, idx);

    CREATE TABLE IF NOT EXISTS beliefs (
        tenant     TEXT NOT NULL,
        user_id    TEXT NOT NULL,
        n_dims     INTEGER NOT NULL,
        mu         BLOB NOT NULL,
        var        BLOB NOT NULL,
        last_seen  BLOB NOT NULL,
        model_ver  TEXT NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (tenant, user_id)
    );

    CREATE TABLE IF NOT EXISTS items (
        tenant           TEXT NOT NULL,
        item_id          TEXT NOT NULL,
        tag_weights      TEXT NOT NULL,
        difficulty_prior REAL,
        attrs            TEXT NOT NULL,
        PRIMARY KEY (tenant, item_id)
    );

    CREATE TABLE IF NOT EXISTS item_params (
        tenant       TEXT NOT NULL,
        item_id      TEXT NOT NULL,
        b            REAL NOT NULL,
        b_var        REAL NOT NULL,
        log_disc     REAL NOT NULL,
        log_disc_var REAL NOT NULL,
        exposure     INTEGER NOT NULL,
        PRIMARY KEY (tenant, item_id)
    );

    CREATE TABLE IF NOT EXISTS signals (
        tenant      TEXT NOT NULL,
        signal_id   TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        item_id     TEXT NOT NULL,
        outcome     REAL NOT NULL,
        ts          REAL NOT NULL,
        propensity  REAL,
        policy_id   TEXT,
        model_ver   TEXT,
        received_at REAL NOT NULL,
        PRIMARY KEY (tenant, signal_id)
    );
    CREATE INDEX IF NOT EXISTS signals_user ON signals(tenant, user_id, ts);
    CREATE INDEX IF NOT EXISTS signals_recv ON signals(tenant, received_at);

    CREATE TABLE IF NOT EXISTS decisions (
        tenant      TEXT NOT NULL,
        decision_id TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        goal        TEXT NOT NULL,
        policy_id   TEXT NOT NULL,
        model_ver   TEXT NOT NULL,
        confidence  TEXT NOT NULL,
        payload     TEXT NOT NULL,
        created_at  REAL NOT NULL,
        PRIMARY KEY (tenant, decision_id)
    );
    CREATE INDEX IF NOT EXISTS decisions_user ON decisions(tenant, user_id, created_at);
    CREATE INDEX IF NOT EXISTS decisions_created ON decisions(tenant, created_at);

    -- Predicted p for each served item, so calibration can be measured ONLINE by
    -- joining against the outcome that arrives later. Without this row the only
    -- calibration number available is the offline one, which cannot detect drift.
    CREATE TABLE IF NOT EXISTS predictions (
        tenant      TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        item_id     TEXT NOT NULL,
        p_hat       REAL NOT NULL,
        decision_id TEXT NOT NULL,
        model_ver   TEXT NOT NULL,
        created_at  REAL NOT NULL,
        PRIMARY KEY (tenant, user_id, item_id)
    );
    CREATE INDEX IF NOT EXISTS predictions_created ON predictions(tenant, created_at);
    """),

    # Inverted index tag -> item. Recall used to be "SELECT ... LIMIT n" with no
    # ORDER BY, which returns an arbitrary and unstable subset: past the limit,
    # most of a catalogue could never be recommended at all, and two identical
    # requests could face different candidate pools. That is a correctness defect,
    # not a scaling one, and it needs an index to fix properly.
    (2, """
    CREATE TABLE IF NOT EXISTS item_tags (
        tenant  TEXT NOT NULL,
        tag     TEXT NOT NULL,
        item_id TEXT NOT NULL,
        weight  REAL NOT NULL,
        PRIMARY KEY (tenant, tag, item_id)
    );
    CREATE INDEX IF NOT EXISTS item_tags_item ON item_tags(tenant, item_id);
    """),

    # Keys in the database rather than only in an env var, so rotation and
    # revocation do not require a process restart.
    (3, """
    CREATE TABLE IF NOT EXISTS api_keys (
        digest     TEXT PRIMARY KEY,
        tenant     TEXT NOT NULL,
        label      TEXT,
        created_at REAL NOT NULL,
        expires_at REAL,
        revoked_at REAL
    );
    CREATE INDEX IF NOT EXISTS api_keys_tenant ON api_keys(tenant);
    """),

    # Tenant-owned L3 policies. The escape hatch was specified in
    # policy.schema.json from the start but had nowhere to live, so the only way
    # to express a utility outside the goal catalogue was to edit server code --
    # an advertised capability that did not exist. Stored per tenant because a
    # policy is customer configuration, not engine configuration.
    (4, """
    CREATE TABLE IF NOT EXISTS policies (
        tenant     TEXT NOT NULL,
        policy_ref TEXT NOT NULL,
        doc        TEXT NOT NULL,
        label      TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (tenant, policy_ref)
    );
    """),

    # A persistent random ordering key. The coverage slice used to be
    # "ORDER BY ((rowid % P) * seed) % P", a computed expression no index can
    # serve: SQLite built a temp B-tree over every one of the tenant's rows on
    # every decide (9ms at 80k items, growing linearly). Storing the permutation
    # instead turns the slice into an index range scan, so the cost follows the
    # slice size rather than the catalogue size.
    #
    # The column, its index and the backfill all live in _post_migrate: unlike
    # every other script here, "ALTER TABLE ADD COLUMN" is not idempotent, so a
    # second process arriving between the version read and this point would die
    # on "duplicate column name" instead of no-opping.
    (5, "SELECT 1;"),

    # A per-tenant catalogue generation counter, bumped on every upsert.
    #
    # Decoding a row into an Item costs two json.loads, and a decide reads ~3200
    # rows -- ~17% of its time spent re-parsing a catalogue that changes rarely.
    # Caching decoded Items in the process is the obvious fix and the obvious
    # trap: under "uvicorn --workers N" one worker's upsert would leave every
    # other worker serving stale items forever, with nothing to notice it. This
    # counter is the cheap cross-process invalidation signal -- one indexed read
    # per query beats 3200 JSON parses, and unlike a TTL it is never wrong.
    (6, """
    CREATE TABLE IF NOT EXISTS catalogue_version (
        tenant  TEXT PRIMARY KEY,
        version INTEGER NOT NULL
    );
    """),
]


SHUFFLE_SPACE = 1 << 62
"""Range of ``items.shuffle_key``. Comfortably inside SQLite's signed 64-bit
INTEGER, so a value is never promoted to REAL and collapsed by rounding."""


def shuffle_key(text: str) -> int:
    """Stable uniform position in ``[0, SHUFFLE_SPACE)`` for a string.

    Explicitly not ``hash()``: that is salted per process, so the same item would
    land somewhere else after every restart and a "reproducible slice" would only
    be reproducible within one process lifetime.
    """
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
    ) % SHUFFLE_SPACE



def _blob(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=np.float64).tobytes()


def _arr(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float64).copy()


@dataclass(frozen=True)
class _RawBelief:
    mu: np.ndarray
    var: np.ndarray
    last_seen: np.ndarray
    model_version: str


@contextmanager
def _null_ctx():
    yield


class SqliteStore:
    def __init__(self, path: str | Path = ":memory:", busy_timeout_ms: int = 30000) -> None:
        self.path = str(path)
        self._shared = self.path == ":memory:"
        self._busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        self._lock = threading.RLock()
        self._shared_con: sqlite3.Connection | None = None
        self._all_cons: list[sqlite3.Connection] = []
        # Decoded-Item cache, keyed (tenant, item_id), validated against the
        # tenant's catalogue_version so another process's upsert cannot leave
        # this one serving stale metadata. See migration 6.
        self._item_cache: dict[tuple[str, str], Item] = {}
        self._cache_gen: dict[str, int] = {}
        # Per tenant, the tag decomposition of each item (see
        # ``MTOR._tag_pairs``). Held here rather than on the per-request item
        # store because the value depends on nothing the request supplies, and
        # invalidated by the same generation counter as the metadata it derives
        # from.
        self._pairs_cache: dict[str, dict] = {}
        # Guards only the invalidation/eviction path, not cache hits. Separate
        # from _lock because that one is held across whole DB transactions in
        # in-memory mode, and a cache purge has no reason to wait behind those.
        self._cache_lock = threading.Lock()
        if self._shared:
            self._shared_con = self._connect()
        self.schema_version = self._migrate()


    # -- connections -------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        con.execute("PRAGMA foreign_keys=ON")
        # Memory-mapped reads and a larger page cache. Measured on a 20k-item
        # catalogue: 55.8ms -> 43.2ms per decide single-threaded, because recall
        # touches far more pages than the 2MB default cache holds.
        #
        # Advisory, not load-bearing: SQLite silently falls back to normal I/O if
        # the mapping fails, and the negative cache_size is a KiB budget rather
        # than a page count so it does not scale with page size. Both are sized
        # per connection, and connections are per thread.
        con.execute("PRAGMA mmap_size=268435456")
        con.execute("PRAGMA cache_size=-16000")
        con.execute("PRAGMA temp_store=MEMORY")
        return con

    @property
    def _con(self) -> sqlite3.Connection:
        if self._shared:
            assert self._shared_con is not None
            return self._shared_con
        con = getattr(self._local, "con", None)
        if con is None:
            con = self._connect()
            self._local.con = con
            # Tracked so ``close()`` can release connections opened by threads
            # that have since exited; thread-local storage alone leaks them.
            with self._lock:
                self._all_cons.append(con)
        return con

    def _guard(self):
        """Only in-memory mode needs a mutex; per-thread connections do not."""
        return self._lock if self._shared else _null_ctx()

    def close(self) -> None:
        with self._lock:
            if self._shared and self._shared_con is not None:
                self._shared_con.close()
                self._shared_con = None
            for con in self._all_cons:
                try:
                    con.close()
                except sqlite3.Error:
                    pass                     # already closed; nothing to release
            self._all_cons.clear()
            if getattr(self._local, "con", None) is not None:
                self._local.con = None


    @contextmanager
    def transaction(self):
        """BEGIN IMMEDIATE: takes the write lock at entry, so a read-modify-write
        cannot interleave with another writer even across processes."""
        con = self._con
        with self._guard():
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
            except BaseException:
                con.rollback()
                raise
            else:
                con.commit()

    # -- migration ---------------------------------------------------------

    def _migrate(self) -> int:
        con = self._con
        # Locked unconditionally, not only in shared mode. Two threads
        # constructing a file-backed store concurrently would otherwise both run
        # the loop; the DDL is idempotent but the ``schema_version`` insert is not,
        # and the loser died on a primary-key conflict inside __init__.
        with self._lock:
            con.execute("CREATE TABLE IF NOT EXISTS schema_version ("
                        "version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)")
            con.commit()
            row = con.execute("SELECT MAX(version) v FROM schema_version").fetchone()
            current = row["v"] or 0
            for version, script in MIGRATIONS:
                if version <= current:
                    continue
                con.executescript(script)
                self._post_migrate(con, version)
                # OR IGNORE covers another *process* having applied the same
                # version between the read above and here.
                con.execute("INSERT OR IGNORE INTO schema_version(version, applied_at) "
                            "VALUES (?,?)", (version, time.time()))
                con.commit()
                current = version
            return int(current)


    @staticmethod
    def _post_migrate(con: sqlite3.Connection, version: int) -> None:
        """Data backfill that a DDL script cannot express.

        Migration 2 adds the tag index; an existing deployment already has items,
        so the index has to be built from them. Without this step the upgrade
        would appear to succeed and then recall nothing.

        Migration 5 adds ``items.shuffle_key``. Same reasoning: existing rows
        would all sit at the default 0, and a permutation where every item shares
        one key is not a permutation -- the coverage slice would degrade to
        item_id order and stop being a slice at all.
        """
        if version == 2:
            rows = con.execute("SELECT tenant, item_id, tag_weights FROM items").fetchall()
            payload = []
            for r in rows:
                for tag, w in (json.loads(r["tag_weights"]) or {}).items():
                    payload.append((r["tenant"], tag, r["item_id"], float(w)))
            if payload:
                con.executemany(
                    "INSERT OR REPLACE INTO item_tags(tenant,tag,item_id,weight) "
                    "VALUES (?,?,?,?)", payload)
            return

        if version == 5:
            cols = {r["name"] for r in con.execute("PRAGMA table_info(items)").fetchall()}
            if "shuffle_key" not in cols:
                con.execute("ALTER TABLE items ADD COLUMN shuffle_key "
                            "INTEGER NOT NULL DEFAULT 0")
            con.execute("CREATE INDEX IF NOT EXISTS items_shuffle "
                        "ON items(tenant, shuffle_key, item_id)")
            rows = con.execute("SELECT tenant, item_id FROM items").fetchall()
            if rows:
                con.executemany(
                    "UPDATE items SET shuffle_key = ? WHERE tenant = ? AND item_id = ?",
                    [(shuffle_key(r["item_id"]), r["tenant"], r["item_id"]) for r in rows])
            return

    # -- tag space ---------------------------------------------------------

    def tag_space(self, tenant: str) -> TagSpace:
        with self._guard():
            rows = self._con.execute(
                "SELECT tag, idx FROM tags WHERE tenant = ? ORDER BY idx", (tenant,)
            ).fetchall()
        if not rows:
            return TagSpace(index_of={}, tag_of=[])
        return TagSpace(index_of={r["tag"]: r["idx"] for r in rows},
                        tag_of=[r["tag"] for r in rows])

    def ensure_tags(self, tenant: str, tags: Iterable[str]) -> TagSpace:
        """Register unseen tags and return the resulting space.

        The reserved latent dimension is created first for every tenant, so an
        item with no tags at all still has somewhere to land. Requiring a
        taxonomy up front would be an adoption blocker, so the space grows on
        first sight instead.
        """
        wanted = [TagSpace.LATENT] + [t for t in dict.fromkeys(tags) if t != TagSpace.LATENT]
        with self.transaction() as con:
            rows = con.execute("SELECT tag, idx FROM tags WHERE tenant = ?", (tenant,)).fetchall()
            index = {r["tag"]: r["idx"] for r in rows}
            nxt = max(index.values()) + 1 if index else 0
            for t in wanted:
                if t in index:
                    continue
                con.execute("INSERT INTO tags(tenant, tag, idx) VALUES (?,?,?)", (tenant, t, nxt))
                index[t] = nxt
                nxt += 1
        return self.tag_space(tenant)

    # -- beliefs -----------------------------------------------------------

    def load_raw_belief(self, tenant: str, user_id: str, con=None) -> _RawBelief | None:
        cur = con if con is not None else self._con
        with self._guard() if con is None else _null_ctx():
            row = cur.execute(
                "SELECT n_dims, mu, var, last_seen, model_ver FROM beliefs "
                "WHERE tenant = ? AND user_id = ?", (tenant, user_id)
            ).fetchone()
        if row is None:
            return None
        return _RawBelief(mu=_arr(row["mu"]), var=_arr(row["var"]),
                          last_seen=_arr(row["last_seen"]), model_version=row["model_ver"])

    def save_belief(self, tenant: str, belief: Belief, now: float, con=None) -> None:
        sql = ("INSERT INTO beliefs(tenant,user_id,n_dims,mu,var,last_seen,model_ver,updated_at) "
               "VALUES (?,?,?,?,?,?,?,?) "
               "ON CONFLICT(tenant,user_id) DO UPDATE SET "
               "n_dims=excluded.n_dims, mu=excluded.mu, var=excluded.var, "
               "last_seen=excluded.last_seen, model_ver=excluded.model_ver, "
               "updated_at=excluded.updated_at")
        args = (tenant, belief.user_id, int(belief.mu.size), _blob(belief.mu),
                _blob(belief.var), _blob(belief.last_seen), belief.model_version, now)
        if con is not None:
            con.execute(sql, args)
            return
        with self.transaction() as c:
            c.execute(sql, args)

    # -- items -------------------------------------------------------------

    def upsert_items(self, tenant: str, items: Sequence[Item]) -> dict[str, int]:
        """Register or update items; returns created/updated counts.

        The split is reported because "did my 5000-item push create rows or
        overwrite them?" is otherwise unanswerable from the outside, and silence
        there is how a caller discovers an id collision months later.
        """
        ids = [it.id for it in items]
        with self.transaction() as con:
            existing: set[str] = set()
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                existing.update(r["item_id"] for r in con.execute(
                    f"SELECT item_id FROM items WHERE tenant = ? AND item_id IN ({marks})",
                    (tenant, *chunk)).fetchall())
            for it in items:
                con.execute(
                    "INSERT INTO items(tenant,item_id,tag_weights,difficulty_prior,attrs,"
                    "shuffle_key) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(tenant,item_id) DO UPDATE SET "
                    "tag_weights=excluded.tag_weights, "
                    "difficulty_prior=excluded.difficulty_prior, attrs=excluded.attrs",
                    (tenant, it.id, json.dumps(dict(it.tag_weights)),
                     it.difficulty_prior, json.dumps(dict(it.attrs)),
                     shuffle_key(it.id)),
                )
                # Re-tagging must remove stale edges, or an item keeps being
                # recalled under a tag it no longer carries.
                con.execute("DELETE FROM item_tags WHERE tenant = ? AND item_id = ?",
                            (tenant, it.id))
                if it.tag_weights:
                    con.executemany(
                        "INSERT INTO item_tags(tenant,tag,item_id,weight) VALUES (?,?,?,?)",
                        [(tenant, t, it.id, float(w)) for t, w in it.tag_weights.items()])
            # Inside the same transaction as the writes: a reader that sees the
            # new rows must also see the new generation, or it would cache the
            # fresh row under the old generation and then never re-read it.
            con.execute("INSERT INTO catalogue_version(tenant, version) VALUES (?, 1) "
                        "ON CONFLICT(tenant) DO UPDATE SET version = version + 1",
                        (tenant,))
        updated = len(set(ids) & existing)
        return {"total": len(items), "created": len(set(ids)) - updated, "updated": updated}

    def list_items(self, tenant: str, after: str | None = None,
                   limit: int = 200) -> list[Item]:
        """A page of the catalogue, keyset-paginated by item id.

        Keyset rather than OFFSET so a page cannot skip or repeat rows while the
        catalogue is being written to, and so paging a large catalogue stays O(1)
        per page.
        """
        with self._guard():
            gen = self._catalogue_gen(tenant)
            if after:
                rows = self._con.execute(
                    "SELECT item_id, tag_weights, difficulty_prior, attrs FROM items "
                    "WHERE tenant = ? AND item_id > ? ORDER BY item_id LIMIT ?",
                    (tenant, after, limit)).fetchall()
            else:
                rows = self._con.execute(
                    "SELECT item_id, tag_weights, difficulty_prior, attrs FROM items "
                    "WHERE tenant = ? ORDER BY item_id LIMIT ?", (tenant, limit)).fetchall()
        return self._items_from(tenant, rows, gen)


    def get_items(self, tenant: str, item_ids: Sequence[str] | None = None,
                  limit: int = 5000) -> list[Item]:
        """Fetch by id, or a deterministically ordered page of the catalogue.

        The ordering matters: an unordered LIMIT returns an arbitrary subset that
        can differ between two identical calls. Prefer ``recall`` for candidate
        generation -- this path is for explicit id lists and diagnostics.

        With ids, the result follows the caller's order rather than whatever order
        the ``IN`` produced, and comes from the metadata cache when it is warm.
        """
        if item_ids is not None:
            with self._guard():
                gen = self._catalogue_gen(tenant)
            return self._items_by_ids(tenant, list(item_ids), gen)
        with self._guard():
            gen = self._catalogue_gen(tenant)
            rows = self._con.execute(
                "SELECT item_id, tag_weights, difficulty_prior, attrs FROM items "
                "WHERE tenant = ? ORDER BY item_id LIMIT ?", (tenant, limit)).fetchall()
        return self._items_from(tenant, rows, gen)

    @staticmethod
    def _row_to_item(r: sqlite3.Row) -> Item:
        return Item(id=r["item_id"], tag_weights=json.loads(r["tag_weights"]),
                    difficulty_prior=r["difficulty_prior"], attrs=json.loads(r["attrs"]))

    ITEM_CACHE_MAX = 100_000
    """Cap on cached decoded Items. At ~900 B each that is well under 100 MB, and
    it is a whole-cache-drop rather than an LRU on purpose: tracking recency would
    need bookkeeping on the read path, which is the path being optimised."""

    def _catalogue_gen(self, tenant: str) -> int:
        """The tenant's catalogue generation. Read *before* the SELECT it guards.

        Order matters. If the generation were read after the rows, an upsert
        committing in between would let this file fresh-looking rows under the new
        generation while the rows themselves came from the old snapshot -- stale
        data that nothing would ever evict. Reading it first can only err the
        other way: rows newer than their label, which the next call discards.
        """
        row = self._con.execute(
            "SELECT version FROM catalogue_version WHERE tenant = ?", (tenant,)).fetchone()
        return int(row["version"]) if row else 0

    def _items_from(self, tenant: str, rows: Sequence[sqlite3.Row], gen: int) -> list[Item]:
        """Decode rows into Items, reusing anything already decoded.

        The generation check is what makes this safe across processes: a sibling
        worker's upsert bumps ``catalogue_version``, and this drops the tenant's
        entries rather than serving what it parsed before the change.

        Endpoints are sync ``def``, so Starlette runs them on a threadpool and
        several threads share this cache. The purge therefore has to be atomic
        with respect to the flag that says it happened: publishing the new
        generation *before* finishing the purge would let a second thread skip
        invalidation and then read an entry the first thread had not yet removed.
        Hence double-checked locking, with the flag set last.
        """
        if not rows:
            return []
        self._expire(tenant, gen)
        cache = self._item_cache
        out: list[Item] = []
        for r in rows:
            key = (tenant, r["item_id"])
            item = cache.get(key)
            if item is None:
                item = self._row_to_item(r)
                cache[key] = item
            out.append(item)
        return out

    def _expire(self, tenant: str, gen: int) -> None:
        """Drop the tenant's cached metadata if its catalogue has moved on."""
        if self._cache_gen.get(tenant) == gen:
            return
        with self._cache_lock:
            if self._cache_gen.get(tenant) == gen:
                return
            for key in [k for k in self._item_cache if k[0] == tenant]:
                self._item_cache.pop(key, None)
            # Derived from the metadata just dropped, so it expires with it.
            self._pairs_cache.pop(tenant, None)
            if len(self._item_cache) > self.ITEM_CACHE_MAX:
                self._item_cache.clear()
                self._pairs_cache.clear()
                self._cache_gen.clear()
            self._cache_gen[tenant] = gen

    def _items_by_ids(self, tenant: str, ids: Sequence[str], gen: int) -> list[Item]:
        """Resolve ids in order, reading metadata only for what is not cached.

        The recall paths know which items they want before they know anything
        about them, and on a warm catalogue the answer is already decoded here.
        Selecting the two JSON blobs anyway meant SQLite read, and this process
        re-parsed, ~2000 rows per decide to reproduce objects it was holding.

        Ids that resolve to nothing are dropped rather than raising: the id came
        from ``item_tags``, and while nothing in this schema deletes an item, a
        caller-supplied list is not owed the assumption.
        """
        if not ids:
            return []
        self._expire(tenant, gen)
        cache = self._item_cache
        missing = [i for i in ids if (tenant, i) not in cache]
        if missing:
            with self._guard():
                for i in range(0, len(missing), 500):    # SQLite parameter cap
                    chunk = missing[i:i + 500]
                    marks = ",".join("?" * len(chunk))
                    for r in self._con.execute(
                            f"SELECT item_id, tag_weights, difficulty_prior, attrs "
                            f"FROM items WHERE tenant = ? AND item_id IN ({marks})",
                            (tenant, *chunk)).fetchall():
                        cache[(tenant, r["item_id"])] = self._row_to_item(r)
        out = []
        for i in ids:
            item = cache.get((tenant, i))
            if item is not None:
                out.append(item)
        return out


    def tag_pairs_cache(self, tenant: str) -> dict:
        """The tenant's slot in the tag-decomposition memo, created on demand.

        Handed out by reference and mutated by whoever holds it, which is safe for
        the same reason the Item cache is: keys are per tenant, values are
        immutable tuples, and every read of an item passes ``_items_from`` first,
        so a generation bump has already emptied this slot before anything can
        read a decomposition of metadata that has since changed.
        """
        slot = self._pairs_cache.get(tenant)
        if slot is None:
            with self._cache_lock:
                slot = self._pairs_cache.setdefault(tenant, {})
        return slot



    def item_count(self, tenant: str) -> int:
        with self._guard():
            return int(self._con.execute(
                "SELECT COUNT(*) c FROM items WHERE tenant = ?", (tenant,)).fetchone()["c"])

    def recall_by_tags(self, tenant: str, tags: Sequence[str], limit: int) -> list[Item]:
        """Items carrying any of ``tags``, heaviest tag weight first.

        Ordering by weight is the point: an item that is mostly about a tag we
        care about is a better candidate than one that merely mentions it, and the
        ordering has to be deterministic so the same request sees the same pool.

        Aggregate first, join second. The obvious phrasing joins ``items`` before
        the ``GROUP BY``, which decodes the two JSON blobs of every item that
        merely mentions any requested tag -- on a 20k catalogue that is ~20k row
        reads to keep 2k, and it measured as ~46% of a decide. The inner query
        below touches only the ``item_tags`` index, so ``items`` is read exactly
        once per row that survives the LIMIT.

        Then the join goes too: what this needs from ``items`` is metadata it has
        usually already decoded, so it takes ids here and lets ``_items_by_ids``
        read only the ones it is missing. The ordering is the inner query's, which
        is what the outer ``ORDER BY`` was restating.
        """
        if not tags or limit <= 0:
            return []
        marks = ",".join("?" * len(tags))
        with self._guard():
            gen = self._catalogue_gen(tenant)
            rows = self._con.execute(
                f"SELECT item_id, MAX(weight) AS w FROM item_tags "
                f"WHERE tenant = ? AND tag IN ({marks}) "
                f"GROUP BY item_id "
                f"ORDER BY w DESC, item_id ASC LIMIT ?",
                (tenant, *tags, limit)).fetchall()
        return self._items_by_ids(tenant, [r["item_id"] for r in rows], gen)

    def sample_items(self, tenant: str, limit: int, seed: int = 0,
                     exclude: Sequence[str] = ()) -> list[Item]:
        """A deterministic pseudo-random slice of the catalogue.

        Needed because objective-driven recall alone is a closed loop: it can only
        ever surface what the current belief already favours, so nothing outside
        that neighbourhood accumulates exposure and the model can never learn it
        was wrong. The slice is seeded and reproducible, so a decision can be
        replayed.

        Mechanism: ``items.shuffle_key`` holds a fixed random permutation of the
        catalogue (see ``shuffle_key``); the seed picks a start point and this
        reads the window of ``limit`` items from there, wrapping past the end. The
        previous phrasing computed the permutation in the ORDER BY, which no index
        can serve -- SQLite sorted the tenant's entire catalogue on every decide.
        This walks the ``items_shuffle`` index instead, so the work scales with
        the slice rather than the catalogue.

        The permutation is now fixed rather than per-seed, so two seeds slide a
        window over the same order instead of reshuffling. Expected overlap
        between two slices is unchanged (~limit^2/catalogue for uniformly spread
        start points), and every item is still reachable, which is what the
        coverage argument above actually requires.

        A long ``exclude`` list is applied in Python rather than truncated. It used
        to be cut at 500 ids, so a large objective slice let this method re-return
        items the caller had already taken -- the pool then held the same id twice.
        """
        if limit <= 0:
            return []
        excluded = set(exclude)
        clause = ""
        sql_ex: list[str] = []
        if excluded:
            # Bind what fits well inside SQLite's parameter limit; anything past
            # that is filtered below, with the fetch widened to compensate.
            sql_ex = list(excluded)[:500]
            clause = f" AND item_id NOT IN ({','.join('?' * len(sql_ex))})"
        overflow = len(excluded) - len(sql_ex)
        fetch = limit + overflow
        start = shuffle_key(str(seed))

        sql = ("SELECT item_id FROM items "
               f"WHERE tenant = ? AND shuffle_key {{cmp}} ?{clause} "
               "ORDER BY shuffle_key, item_id LIMIT ?")
        with self._guard():
            gen = self._catalogue_gen(tenant)
            rows = self._con.execute(sql.format(cmp=">="),
                                     (tenant, start, *sql_ex, fetch)).fetchall()
            if len(rows) < fetch:
                # Wrap: the window ran off the end of the key space.
                rows = rows + self._con.execute(
                    sql.format(cmp="<"),
                    (tenant, start, *sql_ex, fetch - len(rows))).fetchall()
        out = self._items_by_ids(
            tenant, [r[0] for r in rows if r[0] not in excluded], gen)
        return out[:limit]



    def load_item_params(self, tenant: str, item_ids: Sequence[str], con=None) -> dict[str, dict]:
        cur = con if con is not None else self._con
        out: dict[str, dict] = {}
        ids = list(item_ids)
        with self._guard() if con is None else _null_ctx():
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for r in cur.execute(
                    f"SELECT item_id,b,b_var,log_disc,log_disc_var,exposure FROM item_params "
                    f"WHERE tenant = ? AND item_id IN ({marks})", (tenant, *chunk)
                ).fetchall():
                    out[r["item_id"]] = dict(r)
        return out

    def save_item_params(self, tenant: str, params: Mapping[str, dict], con=None) -> None:
        sql = ("INSERT INTO item_params(tenant,item_id,b,b_var,log_disc,log_disc_var,exposure) "
               "VALUES (?,?,?,?,?,?,?) "
               "ON CONFLICT(tenant,item_id) DO UPDATE SET "
               "b=excluded.b, b_var=excluded.b_var, log_disc=excluded.log_disc, "
               "log_disc_var=excluded.log_disc_var, exposure=excluded.exposure")
        rows = [(tenant, iid, p["b"], p["b_var"], p["log_disc"], p["log_disc_var"],
                 int(p["exposure"])) for iid, p in params.items()]
        if not rows:
            return
        if con is not None:
            con.executemany(sql, rows)
            return
        with self.transaction() as c:
            c.executemany(sql, rows)

    # -- signals (idempotency) --------------------------------------------

    def claim_signal(self, tenant: str, signal_id: str, user_id: str, item_id: str,
                     outcome: float, ts: float, propensity: float | None,
                     policy_id: str | None, model_ver: str | None,
                     now: float, con) -> bool:
        """Insert the signal id; return False if it was already present.

        Idempotency has to be decided inside the same transaction that applies
        the update, otherwise a retry can be admitted while the first attempt is
        still in flight and the observation is counted twice.
        """
        cur = con.execute(
            "INSERT OR IGNORE INTO signals(tenant,signal_id,user_id,item_id,outcome,ts,"
            "propensity,policy_id,model_ver,received_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (tenant, signal_id, user_id, item_id, outcome, ts, propensity,
             policy_id, model_ver, now),
        )
        return cur.rowcount == 1

    def signals_for_ope(self, tenant: str, limit: int = 100000) -> list[dict]:
        with self._guard():
            rows = self._con.execute(
                "SELECT user_id,item_id,outcome,ts,propensity,policy_id FROM signals "
                "WHERE tenant = ? AND propensity IS NOT NULL ORDER BY ts LIMIT ?",
                (tenant, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- decisions (audit) -------------------------------------------------

    def log_decision(self, tenant: str, decision_id: str, user_id: str, goal: str,
                     policy_id: str, model_ver: str, confidence: str,
                     payload: Mapping[str, Any], now: float, con=None) -> None:
        sql = ("INSERT OR REPLACE INTO decisions(tenant,decision_id,user_id,goal,policy_id,"
               "model_ver,confidence,payload,created_at) VALUES (?,?,?,?,?,?,?,?,?)")
        args = (tenant, decision_id, user_id, goal, policy_id, model_ver, confidence,
                json.dumps(payload, ensure_ascii=False), now)
        if con is not None:
            con.execute(sql, args)
            return
        with self.transaction() as c:
            c.execute(sql, args)

    def get_decision(self, tenant: str, decision_id: str, con=None) -> dict[str, Any] | None:
        """One decision, payload parsed.

        Exists so a caller does not have to keep its own copy of what was served.
        The propensity of every returned item is in here, which is the number
        off-policy correction cannot be done without and cannot be reconstructed
        after the fact -- making the client store it was pushing the one piece of
        bookkeeping that must not be lost onto the party least able to keep it.
        """
        cur = con if con is not None else self._con
        with self._guard() if con is None else _null_ctx():
            row = cur.execute(
                "SELECT decision_id,user_id,goal,policy_id,model_ver,confidence,payload,"
                "created_at FROM decisions WHERE tenant = ? AND decision_id = ?",
                (tenant, decision_id)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["payload"] = json.loads(out["payload"])
        return out

    def propensities_for(self, tenant: str, decision_ids: Sequence[str],
                         con=None) -> dict[str, dict[str, float]]:
        """``{decision_id: {item_id: propensity}}`` for a batch of decisions.

        Batched because the alternative -- one lookup per signal -- turns a
        5000-signal ingestion into 5000 extra queries.
        """
        ids = sorted({d for d in decision_ids if d})
        if not ids:
            return {}
        cur = con if con is not None else self._con
        out: dict[str, dict[str, float]] = {}
        with self._guard() if con is None else _null_ctx():
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for r in cur.execute(
                    f"SELECT decision_id, payload FROM decisions "
                    f"WHERE tenant = ? AND decision_id IN ({marks})",
                    (tenant, *chunk)).fetchall():
                    payload = json.loads(r["payload"])
                    out[r["decision_id"]] = {
                        c["item_id"]: c.get("propensity")
                        for c in payload.get("chosen") or []
                        if c.get("propensity") is not None
                    }
        return out

    # -- policies (L3) -----------------------------------------------------

    def save_policy(self, tenant: str, policy_ref: str, doc: Mapping[str, Any],
                    label: str | None, now: float) -> None:
        with self.transaction() as con:
            con.execute(
                "INSERT INTO policies(tenant,policy_ref,doc,label,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(tenant,policy_ref) DO UPDATE SET "
                "doc=excluded.doc, label=excluded.label, updated_at=excluded.updated_at",
                (tenant, policy_ref, json.dumps(doc, ensure_ascii=False), label, now, now))

    def get_policy(self, tenant: str, policy_ref: str) -> dict[str, Any] | None:
        with self._guard():
            row = self._con.execute(
                "SELECT policy_ref,doc,label,created_at,updated_at FROM policies "
                "WHERE tenant = ? AND policy_ref = ?", (tenant, policy_ref)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["doc"] = json.loads(out["doc"])
        return out

    def list_policies(self, tenant: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._guard():
            rows = self._con.execute(
                "SELECT policy_ref,label,created_at,updated_at FROM policies "
                "WHERE tenant = ? ORDER BY policy_ref LIMIT ?", (tenant, limit)).fetchall()
        return [dict(r) for r in rows]

    def delete_policy(self, tenant: str, policy_ref: str) -> bool:
        with self.transaction() as con:
            return con.execute("DELETE FROM policies WHERE tenant = ? AND policy_ref = ?",
                               (tenant, policy_ref)).rowcount == 1

    # -- predictions (online calibration) ---------------------------------


    def log_predictions(self, tenant: str, user_id: str, decision_id: str,
                        model_ver: str, preds: Mapping[str, float], now: float,
                        con=None) -> None:
        sql = ("INSERT INTO predictions(tenant,user_id,item_id,p_hat,decision_id,"
               "model_ver,created_at) VALUES (?,?,?,?,?,?,?) "
               "ON CONFLICT(tenant,user_id,item_id) DO UPDATE SET "
               "p_hat=excluded.p_hat, decision_id=excluded.decision_id, "
               "model_ver=excluded.model_ver, created_at=excluded.created_at")
        rows = [(tenant, user_id, iid, float(p), decision_id, model_ver, now)
                for iid, p in preds.items()]
        if not rows:
            return
        if con is not None:
            con.executemany(sql, rows)
            return
        with self.transaction() as c:
            c.executemany(sql, rows)

    def take_prediction(self, tenant: str, user_id: str, item_id: str, con) -> float | None:
        """Read and consume the stored prediction for a served item, inside the
        observe transaction. Consuming it means each served prediction scores at
        most one outcome -- no double counting when an item is re-served."""
        row = con.execute(
            "SELECT p_hat FROM predictions WHERE tenant=? AND user_id=? AND item_id=?",
            (tenant, user_id, item_id)).fetchone()
        if row is None:
            return None
        con.execute("DELETE FROM predictions WHERE tenant=? AND user_id=? AND item_id=?",
                    (tenant, user_id, item_id))
        return float(row["p_hat"])

    # -- retention ---------------------------------------------------------

    def purge(self, tenant: str, now: float, prediction_ttl_days: float = 30.0,
              signal_ttl_days: float = 400.0,
              decision_ttl_days: float = 90.0,
              dry_run: bool = False) -> dict[str, int]:
        """Bound the append-only tables.

        Defaults are operational choices, not tuned values, and they are set by
        what each table is *for*:

        * predictions -- an unmatched prediction is one that was served and never
          reported. Past a month it will not be reported, and keeping it only
          risks matching a much later interaction to a stale estimate.
        * decisions -- audit trail; a quarter covers "why did it do that?".
        * signals -- the training record and the propensity log, so this is the
          longest. Deleting a signal does not un-learn it (the belief already
          absorbed it), but it does destroy the ability to re-derive or re-evaluate
          off-policy, so shortening this is a real capability loss.

        ``dry_run`` counts what would go without deleting it. The operation is
        irreversible and destroys the audit trail, so being able to see the blast
        radius first is the difference between a retention policy and an accident.
        """
        day = 86400.0
        cutoffs = (
            ("predictions", "predictions", "created_at", now - prediction_ttl_days * day),
            ("decisions", "decisions", "created_at", now - decision_ttl_days * day),
            ("signals", "signals", "received_at", now - signal_ttl_days * day),
        )
        out: dict[str, int] = {}
        if dry_run:
            with self._guard():
                for label, table, col, cutoff in cutoffs:
                    row = self._con.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE tenant = ? AND {col} < ?",
                        (tenant, cutoff)).fetchone()
                    out[label] = int(row[0]) if row else 0
            return out
        with self.transaction() as con:
            for label, table, col, cutoff in cutoffs:
                out[label] = con.execute(
                    f"DELETE FROM {table} WHERE tenant = ? AND {col} < ?",
                    (tenant, cutoff)).rowcount
        return {k: max(v, 0) for k, v in out.items()}


    # -- subject rights ----------------------------------------------------

    def export_user(self, tenant: str, user_id: str) -> dict[str, Any]:
        """Everything held about one user, for a data-access request."""
        with self._guard():
            con = self._con
            belief = con.execute(
                "SELECT n_dims, model_ver, updated_at FROM beliefs "
                "WHERE tenant=? AND user_id=?", (tenant, user_id)).fetchone()
            sigs = con.execute(
                "SELECT signal_id,item_id,outcome,ts,propensity,policy_id,received_at "
                "FROM signals WHERE tenant=? AND user_id=? ORDER BY ts",
                (tenant, user_id)).fetchall()
            decs = con.execute(
                "SELECT decision_id,goal,policy_id,model_ver,confidence,payload,created_at "
                "FROM decisions WHERE tenant=? AND user_id=? ORDER BY created_at",
                (tenant, user_id)).fetchall()
        return {
            "user_id": user_id,
            "belief": dict(belief) if belief else None,
            "signals": [dict(r) for r in sigs],
            "decisions": [dict(r) for r in decs],
        }

    def delete_user(self, tenant: str, user_id: str) -> dict[str, int]:
        """Erase a user.

        Honest limitation, and it must be stated rather than buried: item-side
        parameters (difficulty, slope) are aggregates over all users and are not
        reverted here. They contain no personal data and cannot be attributed to
        an individual, but this is deletion of the subject's record, not
        unlearning of their statistical contribution.
        """
        out = {}
        with self.transaction() as con:
            for table in ("beliefs", "signals", "decisions", "predictions"):
                out[table] = con.execute(
                    f"DELETE FROM {table} WHERE tenant = ? AND user_id = ?",
                    (tenant, user_id)).rowcount
        return {k: max(v, 0) for k, v in out.items()}

    # -- api keys ----------------------------------------------------------

    def add_api_key(self, digest: str, tenant: str, label: str | None = None,
                    now: float | None = None, expires_at: float | None = None) -> None:
        with self.transaction() as con:
            con.execute(
                "INSERT OR REPLACE INTO api_keys(digest,tenant,label,created_at,expires_at,"
                "revoked_at) VALUES (?,?,?,?,?,NULL)",
                (digest, tenant, label, now if now is not None else time.time(), expires_at))

    def revoke_api_key(self, digest: str, now: float | None = None) -> bool:
        with self.transaction() as con:
            cur = con.execute("UPDATE api_keys SET revoked_at = ? WHERE digest = ? "
                              "AND revoked_at IS NULL",
                              (now if now is not None else time.time(), digest))
            return cur.rowcount == 1

    def tenant_for_key(self, digest: str, now: float | None = None) -> str | None:
        t = now if now is not None else time.time()
        with self._guard():
            row = self._con.execute(
                "SELECT tenant FROM api_keys WHERE digest = ? AND revoked_at IS NULL "
                "AND (expires_at IS NULL OR expires_at > ?)", (digest, t)).fetchone()
        return row["tenant"] if row else None

    # -- diagnostics -------------------------------------------------------

    def ping(self) -> None:
        """Touch the database. Used by the readiness probe -- a health endpoint
        that answers without consulting its dependency reports health it has not
        checked."""
        with self._guard():
            self._con.execute("SELECT 1 FROM schema_version LIMIT 1").fetchone()

    def counts(self, tenant: str) -> dict[str, int]:
        with self._guard():
            def q(t):
                return self._con.execute(
                    f"SELECT COUNT(*) c FROM {t} WHERE tenant = ?", (tenant,)).fetchone()["c"]
            return {"tags": q("tags"), "beliefs": q("beliefs"), "items": q("items"),
                    "item_tags": q("item_tags"), "signals": q("signals"),
                    "decisions": q("decisions"), "predictions": q("predictions"),
                    "policies": q("policies")}


class TenantItemStore(ItemStore):
    """``ItemStore`` backed by the database, scoped to one tenant.

    Loads lazily and writes only what changed. Subclassing rather than
    reimplementing keeps the learning maths in exactly one place -- MTOR's
    update path is untouched and cannot drift from the in-memory version used by
    the evaluation harnesses.
    """

    def __init__(self, cfg: MTORConfig, store: SqliteStore, tenant: str, con=None) -> None:
        super().__init__(cfg)
        self.store = store
        self.tenant = tenant
        self.con = con
        self._dirty: set[str] = set()
        self._checked: set[str] = set()
        """Ids a bulk read has already resolved, present or absent.

        Absence has to be remembered, not just presence: an item nobody has
        answered yet has no parameter row, so without this every candidate in
        the pool triggered its own lookup on every request -- one query per
        candidate, which under concurrency cost more than the decision itself.
        """

    @property
    def pairs_cache(self) -> dict:
        """Widen the base class's per-instance memo to the whole process.

        Resolved on every access rather than captured in ``__init__``: a
        concurrent upsert replaces the tenant's slot, and an instance holding the
        old dict would go on reading decompositions of metadata that has already
        been dropped from the Item cache.
        """
        return self.store.tag_pairs_cache(self.tenant)

    def preload(self, item_ids: Sequence[str]) -> None:
        rows = self.store.load_item_params(self.tenant, item_ids, con=self.con)
        self._checked.update(item_ids)
        for iid, p in rows.items():
            self._b[iid] = float(p["b"])
            self._var[iid] = float(p["b_var"])
            self._log_disc[iid] = float(p["log_disc"])
            self._log_disc_var[iid] = float(p["log_disc_var"])
            self._exposure[iid] = int(p["exposure"])

    def ensure(self, item: Item) -> None:
        if item.id in self._b:
            return
        if item.id in self._checked:
            # a bulk read already established this item has no stored row
            super().ensure(item)                 # cold item -> prior from metadata
            self._dirty.add(item.id)
            return
        rows = self.store.load_item_params(self.tenant, [item.id], con=self.con)
        p = rows.get(item.id)
        self._checked.add(item.id)
        if p is None:
            super().ensure(item)                 # cold item -> prior from metadata
            self._dirty.add(item.id)
            return

        self._b[item.id] = float(p["b"])
        self._var[item.id] = float(p["b_var"])
        self._log_disc[item.id] = float(p["log_disc"])
        self._log_disc_var[item.id] = float(p["log_disc_var"])
        self._exposure[item.id] = int(p["exposure"])

    def apply(self, item: Item, delta_b: float, new_var: float) -> None:
        super().apply(item, delta_b, new_var)
        self._dirty.add(item.id)

    def apply_disc(self, item: Item, delta_log: float, new_var: float) -> None:
        super().apply_disc(item, delta_log, new_var)
        self._dirty.add(item.id)

    def flush(self) -> int:
        if not self._dirty:
            return 0
        payload = {iid: {"b": self._b[iid], "b_var": self._var[iid],
                         "log_disc": self._log_disc[iid],
                         "log_disc_var": self._log_disc_var[iid],
                         "exposure": self._exposure[iid]} for iid in self._dirty}
        self.store.save_item_params(self.tenant, payload, con=self.con)
        n = len(self._dirty)
        self._dirty.clear()
        return n
