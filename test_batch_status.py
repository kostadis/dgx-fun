"""Tests for batch_status.compute() reading batch_convert's SQLite state DB."""
import sqlite3
import time

import batch_status as bs


def _make_state(path, root, docs):
    """docs: list of (rel, status, reason). Builds a minimal state DB."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE docs (rel TEXT PRIMARY KEY, status TEXT NOT NULL,
                           reason TEXT DEFAULT '');
    """)
    conn.execute("INSERT INTO meta VALUES('root', ?)", (str(root),))
    conn.execute("INSERT INTO meta VALUES('saved_at', ?)", (str(int(time.time())),))
    conn.executemany("INSERT INTO docs(rel, status, reason) VALUES(?,?,?)", docs)
    conn.commit()
    return conn


def test_counts_from_state_db(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    _make_state(tmp_path / "s.db", root, [
        ("a.pdf", "done", ""),
        ("b.pdf", "pending", ""),
        ("c.pdf", "failed", ""),
        ("d.pdf", "skipped", "library:old_version"),
        ("e.pdf", "skipped", "variant:superseded"),
    ]).close()
    st = bs.compute(tmp_path / "s.db", check_disk=False)
    assert st["target"] == 3          # a, b, c are canonical (non-skipped)
    assert st["converted"] == 1       # a done (status source)
    assert st["failed"] == 1          # c
    assert st["remaining"] == 1       # b
    assert st["skipped"] == 2
    assert st["skip_reasons"]["library:old_version"] == 1
    assert st["skip_reasons"]["variant:superseded"] == 1
    assert st["source"] == "state db status"


def test_check_disk_counts_json_on_disk(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    # 'a' is marked pending but its JSON exists on disk -> counts as converted.
    (root / "a.json").write_text("{}")
    _make_state(tmp_path / "s.db", root, [
        ("a.pdf", "pending", ""),
        ("b.pdf", "pending", ""),
    ]).close()
    st = bs.compute(tmp_path / "s.db", check_disk=True)
    assert st["converted"] == 1       # a.json on disk
    assert st["remaining"] == 1
    assert st["source"] == "json on disk"


def test_reads_during_open_writer(tmp_path):
    """A read-only compute() returns the last committed snapshot while a writer
    holds the DB open with an uncommitted transaction (WAL concurrent read)."""
    root = tmp_path / "lib"
    root.mkdir()
    writer = _make_state(tmp_path / "s.db", root, [("a.pdf", "done", "")])
    writer.execute("PRAGMA journal_mode=WAL")
    # begin an uncommitted write
    writer.execute("BEGIN")
    writer.execute("UPDATE docs SET status='pending' WHERE rel='a.pdf'")
    # reader still sees the committed 'done'
    st = bs.compute(tmp_path / "s.db", check_disk=False)
    assert st["converted"] == 1
    writer.rollback()
    writer.close()
